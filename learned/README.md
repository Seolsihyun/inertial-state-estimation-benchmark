# Small TCN 속도 추정

CF231 실험에서는 Runs 3, 4, 9, 10의 6축 IMU와 GT velocity로 Small TCN을 학습하고, 학습에 사용하지 않은 Run 5에서 평가합니다.

```text
최근 200개의 IMU sample
        ↓
Small TCN (36,003 parameters)
        ↓
heading 좌표계의 속도와 예측 오차 공분산
        ↓
world 좌표계 속도로 변환
        ↓
직접 적분 또는 InEKF velocity update
```

## 파일별 역할

- `small_tcn.py`: network, 입력 정규화, IMU window 생성, 학습과 속도 예측
- `velocity_update.py`: 예측 속도를 InEKF measurement로 넣는 Kalman update
- `cf231_protocol.py`: CF231 CSV 로딩, 시간 동기화, bias 계산과 Run 5 평가 절차
- `runners/run_cf231_small_tcn.py`: 위 과정을 한 번에 실행하고 결과 저장

Run 5의 위치를 미분해 가상 속도를 만드는 방식은 사용하지 않았습니다. 속도 label은 학습 run에서만 만들며, Run 5에서는 처음 정한 상태와 고정 bias 이후에 IMU만 입력합니다.

```bash
pip install -e ".[learned]"
state-estimation-cf231-tcn \
  --dataset data/cf231_leave_one_out/csv \
  --output outputs/cf231_small_tcn
```

현재 저장된 결과는 다음과 같습니다.

| 방법 | 위치 RMSE | SO(3) 자세 RMSE |
|---|---:|---:|
| 고정 bias IMU 적분 | 522.2007 m | 3.5458° |
| Small TCN 속도 적분 | 2.3955 m | 3.5458° |
| Small TCN + InEKF | 2.6131 m | 1.7964° |

Small TCN은 위치 drift를 크게 줄였고, InEKF를 결합하면 위치 RMSE는 조금 증가하지만 자세 RMSE가 감소했습니다. 자세 개선은 InEKF가 학습 속도와 IMU propagation을 covariance에 따라 함께 반영한 결과입니다.

이 방법은 평가할 때 IMU만 사용하지만, 모델을 만들 때 다른 비행의 GT velocity가 필요합니다. 학습이 전혀 없는 IMU-only dead reckoning과 혼동하지 않도록 결과에 학습 기반 방법이라고 표시합니다.
