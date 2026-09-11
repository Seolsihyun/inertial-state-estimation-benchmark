# 실험 재현 절차

## 1. 공통 준비

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[test]"
python -m pytest -q
```

TCN을 실행할 때는 `pip install -e ".[learned]"`를 사용합니다. 데이터는 [DATASETS.md](DATASETS.md)의 구조로 배치합니다.

## 2. 테스트가 확인하는 것

- 모든 필터의 `predict/update/estimate_pose` API
- 초기 sample을 이중 적분하지 않는지
- InEKF right correction과 measurement Jacobian
- RPY에서 계산한 SO(3) 오차
- TCN parameter 수, causal window, normalization shape

테스트 통과는 실제 데이터 성능을 보장하지 않습니다. 식·shape·순서의 회귀를 막는 장치입니다.

## 3. EuRoC 연속 IMU-only

```bash
state-estimation-run-all \
  --config config/euroc_imu_only.yaml \
  --filters ekf ukf pf inekf \
  --particles 500
```

검산:

- 각 `run_manifest.json`의 `resolved_run.position_updates` = 0
- `samples` = 5,824
- 초기 bias가 EuRoC GT 첫 sample과 같음
- `estimate.csv` 첫 행이 GT 첫 행과 같음

## 4. EuRoC GT pseudo-position 결합

```bash
state-estimation-run-all \
  --config config/euroc.yaml \
  --filters ekf ukf pf inekf \
  --particles 500
```

`config/euroc.yaml`의 `imu_stride=5`, `measurement_stride=20`이므로 update는 약 2 Hz입니다. 전체 update 횟수는 291회여야 합니다. 측정은 실제 GNSS가 아니라 GT position에 0.05 m Gaussian noise를 더한 값입니다.

## 5. 3D motion regimes

```bash
state-estimation-motion-benchmark --config config/motion_regimes.yaml
```

각 regime은 45 s, 50 Hz입니다. 이 실험은 회전만 변경하지 않고 궤적 주파수, z 운동, roll/pitch도 함께 바꾸므로 3D motion-complexity stress test로 표기합니다.

## 6. CF231 held-out Run 5

```bash
state-estimation-cf231-tcn \
  --dataset data/cf231_leave_one_out/csv \
  --training-runs 3,4,9,10 \
  --bias-runs 3,9,10 \
  --test-run 5 \
  --window-samples 200 \
  --training-stride 5 \
  --update-stride 10 \
  --downsample 2 \
  --cv-epochs 12 \
  --epochs 32 \
  --seed 42 \
  --output outputs/cf231_small_tcn
```

데이터 분할:

| 용도 | run |
|---|---|
| TCN training | 3, 4, 9, 10 |
| leave-one-run-out covariance | 3, 4, 9, 10 |
| gyro intrinsic / training noise | 3, 9, 10 |
| final test | 5 |

Run 5 GT를 사용하는 곳은 초기 `R,v,p`, 정지 구간 accelerometer bias의 gravity direction, 모든 trajectory가 완성된 후 `score()`입니다. Run 5 GT를 training target, velocity update, position update에 쓰지 않습니다.

## 7. i2Nav/Pohang

```bash
state-estimation-run-all --config config/i2nav_street00.yaml
state-estimation-run-all --config config/pohang05.yaml
```

i2Nav는 F9P GNSS를 update에, `groundtruth.nav`를 evaluation에 사용합니다. Pohang 기본 loader는 `baseline.txt`를 update와 evaluation에 같이 쓰므로 위치 RMSE를 독립 localization 성능으로 보고하지 않습니다.

## 8. 결과 파일

```text
outputs/<dataset>/<sequence>/<filter>/
  estimate.csv
  metrics.json
  run_manifest.json
  runtime.json
  trajectory.png
  error.png
```

결과를 공유할 때는 `metrics.json`만 복사하지 말고 `run_manifest.json`, 사용한 YAML, commit hash를 함께 남깁니다.
