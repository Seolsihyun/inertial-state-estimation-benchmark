# 수식과 코드 대응표

## 1. 표기와 frame

| 기호 | 의미 | 코드 |
|---|---|---|
| `R` | body-to-world rotation | `estimator.Rot` |
| `v` | world velocity | `estimator.v` |
| `p` | world position | `estimator.p` |
| `bg`, `ba` | gyro, accelerometer bias | `gyro_bias`, `accel_bias` |
| `P` | 15D local error covariance | `estimator.P` |
| `Q`, `Rm` | process, measurement covariance | filter config |

InEKF group state는

```text
    [ R  v  p ]
X = [ 0  1  0 ]
    [ 0  0  1 ]
```

이고 error 순서는 `[dtheta, dv, dp, dbg, dba]`입니다.

## 2. nominal propagation

| 수식 | 코드 위치 |
|---|---|
| `omega = gyro - bg`, `f = accel - ba` | `filters/InEKF.py::_propagate_nominal` |
| `R+ = R Exp(omega dt)` | `exp_so3` |
| `v+ = v + (R Gamma1 f + g)dt` | `left_jacobian_so3` |
| `p+ = p + vdt + R Gamma2 f dt^2 + gdt^2/2` | `gamma2_so3` |

EKF/UKF/PF/ESKF도 동일 IMU 입력과 중력 정의를 쓰지만, quaternion additive state, sigma point, particle, error-state로 uncertainty를 다룹니다.

## 3. InEKF covariance propagation

InEKF의 local error state와 `process_noise_diag`의 순서는 같습니다.

```text
delta x = [dtheta, dv, dp, dbg, dba]
```

| `process_noise_diag` slice | 코드에서의 의미 |
|---|---|
| `[0:3]` | local rotation error `dtheta`에 더하는 축별 noise variance weight |
| `[3:6]` | local velocity error `dv`에 더하는 축별 noise variance weight |
| `[6:9]` | local position error `dp`에 더하는 축별 noise variance weight |
| `[9:12]` | gyro-bias error `dbg`에 더하는 축별 noise variance weight |
| `[12:15]` | accelerometer-bias error `dba`에 더하는 축별 noise variance weight |

`process_noise_scale`을 각 원소에 곱한 뒤 `diagonal_covariance()`가 최소 `1e-12`로 clip하여 `Q`를 만듭니다. `predict()`에 구현된 실제 공분산 예측식은 다음과 같습니다.

```text
q = max(process_noise_scale, 0) * process_noise_diag
Q = diag(max(q, 1e-12))
P_raw[k+1] = Phi[k] P[k] Phi[k]^T
             + Phi[k] Q Phi[k]^T * max(dt, 1e-9)
P[k+1] = stabilize(P_raw[k+1])
```

`stabilize()`는 행렬을 대칭화하고 고유값을 설정된 floor/ceiling 범위로 제한합니다. 즉 현재 코드는 일반적인 `P = Phi P Phi^T + G Qc G^T dt`를 별도의 `G`로 구현한 것이 아니라, 위의 `Phi Q Phi^T * dt`를 그대로 사용합니다.

runner가 사용하는 `InEKF`는 `InEKFAnalytic15D`의 alias이며, 기본 `jacobian_mode="analytic"`에서 `_analytic_process_jacobian()`이 discrete transition `Phi`를 만듭니다. 다만 이 함수 안에서 gyro-bias coupling 블록은 3축에 대한 local central finite difference로 구합니다. 검산용 `finite` mode는 전체 15D process Jacobian을 central finite difference로 계산하며, 테스트에서 기본 경로와 비교합니다.

### CF231에서 넣는 process noise

`learned/cf231_protocol.py::training_calibration()`은 기본 bias runs 3, 9, 10의 각 초기 1 s 정지 샘플에서 축별 robust variance `(1.4826 * MAD)^2`를 구하고, 그 run들 간 median을 저장합니다. `runners/run_cf231_small_tcn.py::make_inekf()`는 이 값을 다음과 같이 사용합니다.

```text
process_noise_diag[0:3] = gyro_sample_variance
process_noise_diag[3:6] = accel_sample_variance
process_noise_diag[6:15] = 0
```

마지막 9개 원소의 0은 `diagonal_covariance()`에서 실제 `Q`를 만들 때 `1e-12`로 clip됩니다. 이 `Q`는 fixed-bias IMU DR에서는 nominal trajectory를 바꾸지 않고 공분산만 전파하며, Small TCN + InEKF에서는 velocity update의 Kalman gain에 영향을 줍니다.

Run 5 초기 정지 구간은 이 process-noise variance를 맞추는 데 쓰지 않습니다. `test_calibration()`이 그 구간의 gyro/accelerometer **mean**으로 고정 bias를 구해 propagation IMU를 미리 보정하고, 필터 내부 bias는 0, `update_biases=False`로 실행합니다.

현재 CF231 noise 설정은 정지 샘플의 robust variance를 직접 매핑한 것입니다. Allan variance/deviation로 continuous-time gyro/accelerometer noise density와 bias random walk를 식별한 것이 아니며, `sample_period_s`를 이용한 spectral-density 변환도 하지 않습니다. 따라서 이 값을 엄밀하게 식별된 continuous-time IMU noise parameter로 해석하면 안 됩니다.

## 4. right perturbation과 update

```text
plus_right(X, delta) = X Exp(delta)
minus_right(Y, X)    = Log(X^-1 Y)
```

position residual을 `r = z_p - p` 로 두면 `X Exp(delta)`의 1차 근사에서 `p_plus ~= p + R delta_p`이므로 `H_p = [0, 0, R, 0, 0]`입니다. velocity도 같은 이유로 `H_v = [0, R, 0, 0, 0]`입니다.

Kalman 계산은 `utils/filter_math.py::kalman_update()`, group correction은 `filters/InEKF.py::measurement_update()`와 `velocity_update()`에 있습니다.

## 5. bias state

공통 필터는 15D 상태에 bias error를 포함합니다. 

| 실험 | bias 초기값 | 실행 중 update |
|---|---|---|
| EuRoC | GT 첫 sample | 필터 설정에 따라 공분산 전파/update |
| synthetic | generator의 초기 true bias | random walk는 필터에 알려주지 않음 |
| CF231 fixed/TCN | Run 5 초기 정지 구간에서 추정해 IMU를 사전 보정; 필터 내부 bias는 0 | `update_biases=False`, 고정 |
| i2Nav/Pohang | 0 | 현재 별도 calibration 없음 |

## 6. CF231 Small TCN 경로

```text
Runs 3,4,9,10 IMU window
 -> SmallTCN
 -> heading-frame velocity
 -> InEKF attitude로 world velocity로 회전
 -> velocity residual
 -> InEKF Kalman update
```

- window: 200 samples, downsample 2
- 입력: 6-axis IMU
- 출력: heading frame 3D velocity
- test: Run 5
- Run 5 제외 범위: training, cross-validation covariance
- Run 5 GT 사용: 초기화, 초기 accelerometer bias의 gravity direction, 실행 후 score

코드 순서는 `runners/run_cf231_small_tcn.py` 안의 `training_windows()` -> `fit_small_tcn()` -> `prepare_inference()` -> `propagate()` -> `score()`입니다.

## 7. 평가식

```text
position error_i = ||p_est_i - p_gt_i||_2
position RMSE    = sqrt(mean(position error_i^2))
R_error_i        = R_gt_i^T R_est_i
SO(3) error_i    = acos((trace(R_error_i)-1)/2)
```

Euler 각은 각 축의 원인을 보기 위해 남기고, SO(3) 오차는 자세 전체의 coordinate-free 크기를 보기 위해 함께 계산합니다.
