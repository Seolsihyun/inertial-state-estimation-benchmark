# Small TCN 속도 추정

CF231 실험에서는 Runs 3, 4, 9, 10의 6축 IMU와 GT velocity로 Small TCN을 학습하고, 학습에 사용하지 않은 Run 5에서 평가합니다.

```text
최근 200 raw-sample 구간
(downsample=2 -> 100 time steps)
        ↓
Small TCN (36,003 parameters)
        ↓
heading 좌표계의 3차원 속도
        ↓
world 좌표계 속도로 변환
        ↓
직접 적분 또는 InEKF velocity update
```

InEKF update에 쓰는 예측 오차 공분산은 TCN의 출력이 아니라 training run leave-one-run-out 잔차에서 계산합니다.

## 파일별 역할

- `datasets/cf231.py`: CSV loading, timestamp synchronization, position interpolation, rotation SLERP, GT-derived kinematics
- `learned/cf231_protocol.py`: calibration protocol, stationary segment detection, bias estimation, IMU preprocessing
- `learned/small_tcn.py`: window generation, normalization, training, velocity prediction
- `learned/velocity_update.py`: learned velocity를 InEKF measurement로 사용하는 update
- `runners/run_cf231_small_tcn.py`: 위 과정을 한 번에 실행하고 결과 저장

Run 5의 위치를 미분해 가상 속도를 만드는 방식은 사용하지 않았습니다. Run 5 GT/reference는 초기 position, velocity, orientation과 accelerometer bias의 gravity direction을 정하는 데 사용합니다. 초기화 이후 time-varying Run 5 GT는 propagation, TCN inference, InEKF update에 사용하지 않고, 전체 GT trajectory는 최종 scoring에 사용합니다.

Small TCN 입력은 6축 IMU이며, 각 run의 초기 최대 100개 gyroscope sample 평균을 gyro 3축에서 제거합니다. 이 전처리에는 GT를 사용하지 않습니다.

Small TCN loose integration에서는 TCN이 자세를 추정하지 않습니다. world-frame velocity 변환에 사용하는 yaw와 최종 rotation은 fixed-bias IMU propagation trajectory에서 가져오고, 속도만 TCN prediction으로 대체합니다. 따라서 fixed-bias IMU DR과 loose integration의 SO(3) error는 동일합니다.

```bash
pip install -e ".[learned]"
state-estimation-cf231-tcn \
  --dataset data/cf231_leave_one_out/csv \
  --output outputs/cf231_small_tcn
```

현재 저장된 결과는 다음과 같습니다.

| 방법 | 위치 RMSE | SO(3) 자세 RMSE |
|---|---:|---:|
| 고정 bias IMU 적분 | 522.2007 m | 3.5535° |
| Small TCN 속도 적분 | 2.3955 m | 3.5535° |
| Small TCN + InEKF | 2.6131 m | 1.7964° |

Small TCN은 위치 drift를 크게 줄였고, InEKF를 결합하면 위치 RMSE는 조금 증가하지만 자세 RMSE가 감소했습니다. 자세 개선은 InEKF가 학습 속도와 IMU propagation을 covariance에 따라 함께 반영한 결과입니다.

초기화 이후 runtime inference에서는 IMU만 사용하지만, 모델을 만들 때 다른 비행의 GT velocity가 필요합니다. 학습이 전혀 없는 IMU-only dead reckoning과 혼동하지 않도록 결과에 학습 기반 방법이라고 표시합니다.
