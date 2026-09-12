# 초보자 실행 매뉴얼

이 문서는 **IMU 한 줄이 상태를 바꾸는 과정, bias와 측정 update가 들어가는 위치, 결과 수치를 검산하는 방법**을 순서대로 설명합니다.

## 1. 용어

- IMU: 가속도계 3축과 자이로 3축
- dead reckoning(DR): 외부 위치 측정 없이 이전 상태와 IMU로만 다음 상태를 계산
- bias: 정지해도 0이 아닌 센서의 지속적인 오프셋
- predict: IMU를 적분하는 단계
- measurement update: GNSS 위치나 학습 속도로 예측을 교정하는 단계
- GT: 추정값을 비교하는 기준값. GT를 update에 쓰면 독립 평가가 아니다.

## 2. 상태, 입력, 단위

```text
R  : body -> world 회전
v  : world frame 속도 [m/s]
p  : world frame 위치 [m]
bg : gyro bias [rad/s]
ba : accelerometer bias [m/s^2]
```

공통 IMU 입력은 `[ax, ay, az, gx, gy, gz]`입니다.

loader 출력은 `datasets/common.py` 안의 `CommonDataset`으로 통일됩니다. 새 데이터를 넣을 때는 loader의 축·단위·timestamp를 먼저 검사합니다.

## 3. 한 sample의 처리 순서

`runners/run_filter.py` 안의 `run_filter()`가 실행 흐름입니다.

```text
YAML -> dataset loader -> 필터 초기화
estimate[0] = timestamps[0]의 초기 상태
i = 1 ... N-1:
    predict(IMU[i], dt[i])
    측정이 있는 시각에만 measurement_update()
    estimate[i] 저장
GT와 error 계산 -> CSV/JSON/PNG 저장
```

0번 sample을 다시 적분하지 않는 이유는 초기 상태가 이미 `timestamps[0]`의 값이기 때문입니다. `tests/test_runner_alignment.py`가 이 정렬을 검사합니다.

## 4. IMU predict

```text
omega = gyro_measured - bg
f     = accel_measured - ba
phi   = omega * dt

R_next = R * Exp(phi)
v_next = v + (R * Gamma1(phi) * f + g) * dt
p_next = p + v * dt + R * Gamma2(phi) * f * dt^2 + 0.5 * g * dt^2
```

`f`는 body frame specific force입니다. 코드는 `filters/InEKF.py::_propagate_nominal()`, SO(3) 함수는 `models/Hoon_lie_group_utils.py`에 있습니다.

## 5. bias 추정

### EuRoC

`datasets/euroc.py`는 GT 첫 sample에 포함된 gyro/accelerometer bias를 초기값으로 읽습니다. 따라서 EuRoC IMU-only 결과는 **GT bias를 알고 시작하는 조건**입니다.

### CF231 Run 5

`learned/cf231_protocol.py::test_calibration()`이 처음 정지 구간을 찾고 한 번만 계산합니다.

```text
gyro bias = mean(gyro during stationary interval)
accel bias = mean(accel) - R_imu_to_world^T * (-g_world)
```

Run 5에서 검출된 정지 구간은 2.627 s입니다.

```text
bg = [-0.0003308, -0.0000921,  0.0000360] rad/s
ba = [ 0.4154277,  0.1496042,  0.0077138] m/s^2
```

가속도 bias, 초기 위치·속도·자세에 Run 5 첫 GT를 사용합니다. 정확한 표기는 **GT-assisted initialization + fixed bias + 이후 IMU-only**입니다. 추정한 bias로 propagation IMU를 미리 보정하므로 필터 내부 bias는 0으로 두고, 초기화 후에는 `update_biases=False`로 다시 바꾸지 않습니다.

### CF231 Small TCN 입력 전처리

Small TCN 입력은 6축 IMU입니다.

```text
[ax, ay, az, gx, gy, gz]
```

각 run에 대해 초기 최대 100개의 gyroscope sample 평균을 계산하고, 이 평균을 gyroscope 3축에서 제거합니다.

```text
gyro_network = gyro - mean(gyro[0:100])
```

이 입력 전처리 자체는 GT를 사용하지 않습니다. 다만 학습할 때는 training run의 GT-derived velocity와 kinematics를 target과 window quality 선별에 사용합니다.

window는 최근 200 raw sample 구간을 사용하고 `downsample=2`이므로 실제 network 입력은 100 time step입니다. network 출력은 heading-frame 3D velocity입니다.

Small TCN loose integration에서는 TCN이 자세를 추정하지 않습니다. 자세와 yaw는 fixed-bias IMU propagation trajectory를 그대로 사용하고, 속도만 TCN prediction으로 바꿉니다.

따라서 Fixed-bias IMU DR과 Small TCN loose integration의 SO(3) error가 동일하게 나타납니다.

## 6. InEKF error와 covariance

InEKF는 `R, v, p`를 `SE_2(3)` matrix `X`로 묶고 작은 오차를 15차원으로 둡니다.

```text
delta x = [delta theta, delta v, delta p, delta bg, delta ba]
X_plus  = X @ Exp(delta xi)
```

이 right-perturbation convention에서

```text
H_position[:, delta_p] = R
H_velocity[:, delta_v] = R
```

입니다. correction도 `X @ Exp(delta)`를 씁니다. `tests/test_inekf_convention.py`가 analytic/finite-difference Jacobian과 correction 방향을 확인합니다.

IMU-only에서는 Kalman update가 없으므로 공분산 `P`가 잘 계산되어도 `R, v, p`를 되돌릴 innovation이 없습니다.

## 7. measurement update

```text
r = z - h(x)
S = H P H^T + R_measurement
K = P H^T S^-1
delta = K r
X <- X @ Exp(delta_xi)
P <- covariance update
```

CF231 TCN은 `z_v`를 속도 측정처럼 넣습니다. 속도 noise covariance는 Run 5에서 맞추지 않고 training run 단위 leave-one-run-out residual로 계산합니다.

## 8. 결과 검산 순서

1. `run_manifest.json`의 mode, bias, gravity, update 횟수를 본다.
2. `estimate.csv` 첫 행이 초기 상태와 같은지 본다.
3. IMU-only라면 `position_updates=0`인지 본다.
4. position RMSE, final, max error를 함께 본다.
5. roll/pitch/yaw와 SO(3) geodesic error를 함께 본다.
6. 실행 시간은 같은 장비·sample·particle 조건에서만 비교한다.

## 9. 재현 명령

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test,learned]"
python -m pytest -q

state-estimation-run-all --config config/euroc_imu_only.yaml --particles 500
state-estimation-run-all --config config/euroc.yaml --particles 500

state-estimation-cf231-tcn \
  --dataset data/cf231_leave_one_out/csv \
  --training-runs 3,4,9,10 \
  --bias-runs 3,9,10 \
  --test-run 5 \
  --output outputs/cf231_small_tcn
```

결과가 다르면 commit hash, Python/NumPy/PyTorch version, YAML, PF particle 수·seed, TCN epoch·seed·training run, 데이터 sample 수를 순서대로 비교합니다.

## 10. 새 데이터 체크리스트

- timestamp가 초 단위인가
- gyro가 rad/s인가
- accelerometer가 m/s²인가
- quaternion 순서가 `wxyz`인가 `xyzw`인가
- `R`이 body-to-world인가
- 중력 부호와 accelerometer 출력이 맞는가
- IMU와 GT의 시각을 어떻게 맞췄는가
- position update와 evaluation GT가 같은 신호인가
- bias를 어느 구간에서 구했는가
