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

error-state와 `process_noise_diag`의 순서는 다음과 같습니다.

```text
[dtheta, dv, dp, dbg, dba]
```

| slice | 의미 |
|---|---|
| `[0:3]` | local rotation error `dtheta` noise weight |
| `[3:6]` | local velocity error `dv` noise weight |
| `[6:9]` | local position error `dp` noise weight |
| `[9:12]` | gyro-bias error `dbg` noise weight |
| `[12:15]` | accelerometer-bias error `dba` noise weight |

`process_noise_scale`을 곱한 뒤 `diagonal_covariance()`가 각 원소를 최소 `1e-12`로 clip해 `Q`를 만듭니다. `filters/InEKF.py::predict()`의 실제 covariance prediction은 다음과 같습니다.

```text
q = max(process_noise_scale, 0) * process_noise_diag
Q = diag(max(q, 1e-12))
P_raw[k+1] = Phi[k] P[k] Phi[k]^T
             + Phi[k] Q Phi[k]^T * max(dt, 1e-9)
P[k+1] = stabilize(P_raw[k+1])
```

즉 현재 구현은 별도의 `G Qc G^T` continuous-to-discrete 계산이 아니라 `Phi Q Phi^T * dt`를 그대로 사용합니다. `stabilize()`는 공분산을 대칭화하고 고유값을 설정된 floor/ceiling 범위로 제한합니다.

CF231의 `training_calibration()`은 기본 Runs 3, 9, 10의 초기 1 s 정지 샘플에서 축별 robust variance `(1.4826 * MAD)^2`를 구한 뒤 run 간 median을 저장합니다. `make_inekf()`는 이 값을 다음 블록에 넣습니다.

```text
process_noise_diag[0:3] = gyro_sample_variance
process_noise_diag[3:6] = accel_sample_variance
process_noise_diag[6:15] = 0
```

마지막 9개 0은 `Q`를 만들 때 `1e-12`로 clip됩니다. Run 5 정지 구간은 process-noise variance가 아니라 gyro/accelerometer **mean**으로 고정 bias를 구하는 데 쓰입니다. 그 bias로 propagation IMU를 미리 보정하고 필터 내부 bias는 0, `update_biases=False`로 실행합니다.

이 noise 설정은 Allan variance/deviation로 continuous-time sensor noise density와 bias random walk를 식별한 것이 아니며, `sample_period_s`를 이용한 spectral-density 변환도 하지 않습니다. 따라서 엄밀한 Allan-variance 기반 continuous-time noise identification으로 해석하면 안 됩니다.

`InEKFAnalytic15D._analytic_process_jacobian()`이 `Phi`를 만듭니다. 이 함수 안의 gyro-bias coupling은 3축 local central finite difference로 구하며, 검산용 `finite` mode는 전체 15D process Jacobian을 central finite difference로 계산합니다.

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

- window: 최근 200 raw sample 구간, downsample 2 -> network 입력 100 time steps
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
