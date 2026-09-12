# Inertial State Estimation Benchmark

IMU로 자세·속도·위치를 적분하고, 위치 또는 학습 속도를 결합하는 과정을 재현하기 위한 코드입니다.
필터 수식, bias 사용 방식, GT 사용 범위, 실행 설정과 결과 파일을 서로 연결해 확인할 수 있게 구성했습니다.

## 먼저 볼 문서

1. [초보자 실행 매뉴얼](docs/BEGINNER_MANUAL.md): IMU 데이터가 결과로 나오는 전체 순서
2. [이전 코드의 문제와 수정 내용](docs/CHANGES_FROM_PREVIOUS.md): 어떤 문제를 왜 고쳤는지
3. [수식과 코드 대응표](docs/IMPLEMENTATION.md): propagation, bias, InEKF Jacobian/update
4. [실험 절차](EXPERIMENTS.md): 명령어, 설정, 출력 파일
5. [데이터셋별 결과](RESULTS.md): 재실행한 수치와 해석 범위

## 이번 작업의 핵심

| 항목 | 이전 상태 | 현재 상태 |
|---|---|---|
| 실행 시각 정렬 | 0번 IMU를 적분한 값을 0번 GT와 비교 | 0번은 초기 상태로 저장, 1번부터 적분 |
| InEKF convention | 주석과 correction 표현이 섞여 있었음 | `X @ Exp(delta)` right perturbation으로 통일·테스트 |
| IMU-only 해석 | sliding window 결과를 연속 DR처럼 보일 수 있었음 | 초기화 1회 후 update 0회인 연속 실험만 DR로 표기 |
| bias | 최종 표에 bias 근거가 보이지 않음 | `metrics.json`과 `run_manifest.json`에 초기 bias·중력·설정 저장 |
| 자세 평가 | roll/pitch/yaw 단독 | Euler 오차 + SO(3) geodesic 오차 |
| 수치 근거 | 원 설정이 없는 과거 표까지 혼재 | 현재 코드로 재현한 결과만 유지 |

자세한 근거는 [이전 코드의 문제와 수정 내용](docs/CHANGES_FROM_PREVIOUS.md)에 파일·함수 단위로 적었습니다.

## 실행 방법

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[test]"
python -m pytest -q
python examples/basic_filter_loop.py
```

원시 데이터는 Git에 포함하지 않습니다. [데이터 준비](DATASETS.md)에 맞게 `data/`에 배치한 뒤 EuRoC를 실행합니다.

```bash
state-estimation-run-all --config config/euroc_imu_only.yaml --particles 500
state-estimation-run-all --config config/euroc.yaml --particles 500
```

각 실행은 `outputs/<dataset>/<sequence>/<filter>/`에 다음을 남깁니다.

- `estimate.csv`: 시각별 추정값과 GT
- `metrics.json`: 위치, Euler 자세, SO(3) 자세 오차
- `run_manifest.json`: 설정, bias, 초기 상태, update 횟수, 데이터 출처, project version과 git commit
- `trajectory.png`, `error.png`: 실행 결과 그래프

## 바로 알아야 할 결론

- IMU-only에서 measurement update가 없으면 covariance가 nominal trajectory를 교정하지 않습니다. 동일 propagation을 쓰는 EKF와 InEKF의 궤적은 거의 같습니다.
- EuRoC `V1_01_easy`에서 연속 IMU-only 위치 RMSE는 약 1.13 km로 발산했습니다.
- 같은 sequence에 GT 위치+인위 noise를 2 Hz로 넣으면 EKF/UKF/InEKF 위치 RMSE는 0.11 m 수준입니다.
- CF231 Run 5는 GT-assisted initial state/calibration 후 time-varying GT 없이 IMU로 추론하고, 전체 GT trajectory는 최종 scoring에 사용합니다.

## 코드 구조

```text
config/             실험 조건과 noise
datasets/           원시 데이터 -> 공통 배열
filters/            EKF, UKF, PF, ESKF, InEKF
learned/            CF231 window, Small TCN, velocity update
runners/            실행 순서와 결과 저장
evaluation/         오차 지표와 그래프
results/            재현한 표·그래프의 근거 파일
tests/              시각 정렬, API, InEKF convention, metric 테스트
```

초기 필터 구조는 `andyjaehun/State_Estimation`을 기준으로 시작했습니다. 지난 학기 benchmark와 이번 추가·수정 범위는 [PROVENANCE.md](PROVENANCE.md)에 구분했습니다.
