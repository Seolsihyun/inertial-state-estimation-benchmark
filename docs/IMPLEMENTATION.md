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

```text
delta_x[k+1] ~= Phi[k] delta_x[k] + G n[k]
P[k+1] = Phi P Phi^T + Qd
```

`InEKFAnalytic15D._analytic_process_jacobian()`이 `Phi`를 만듭니다. 개발 중 검산을 위해 같은 class에 central finite-difference 경로를 남겨 두었습니다. measurement Jacobian은 테스트에서 두 경로를 숫자로 비교합니다.

## 4. right perturbation과 update

```text
plus_right(X, delta) = X Exp(delta)
minus_right(Y, X)    = Log(X^-1 Y)
```

position residual을 `r = z_p - p` 로 두면 `X Exp(delta)`의 1차 근사에서 `p_plus ~= p + R delta_p`이므로 `H_p = [0, 0, R, 0, 0]`입니다. velocity도 같은 이유로 `H_v = [0, R, 0, 0, 0]`입니다.

Kalman 계산은 `utils/filter_math.py::kalman_update()`, group correction은 `filters/InEKF.py::measurement_update()`와 `velocity_update()`에 있습니다.

## 5. bias state

공통 필터는 15D 상태에 bias error를 포함합니다. 다만 실험별 정책이 다릅니다.

| 실험 | bias 초기값 | 실행 중 update |
|---|---|---|
| EuRoC | GT 첫 sample | 필터 설정에 따라 공분산 전파/update |
| synthetic | generator의 초기 true bias | random walk는 필터에 알려주지 않음 |
| CF231 fixed/TCN | Run 5 초기 정지 구간 | `update_biases=False`, 고정 |
| i2Nav/Pohang | 0 | 현재 별도 calibration 없음 |

따라서 데이터셋 간 RMSE를 비교할 때는 bias 조건을 같은 것처럼 해석하면 안 됩니다.

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
