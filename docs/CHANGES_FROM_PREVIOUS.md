# 이전 코드의 문제와 수정 내용

비교 기준은 다음 세 단계입니다.

- 지난 학기: `INHA-Artemis/State_Estimation_Benchmark`
- 이번 작업의 출발점: `andyjaehun/State_Estimation`
- 현재 저장소: 공통 실행기, 연속 DR, CF231 held-out 학습 실험을 합친 구조

## 1. 지난 학기 결과와의 차이

지난 학기 저장소는 필터별 benchmark script가 따로 있었고, KAIST VIO/M2DGR의 저장된 영상·그래프를 중심으로 보여주었습니다. 이번에는 다음을 바꾸었습니다.

| 항목 | 지난 학기 | 현재 |
|---|---|---|
| 실행 구조 | 필터·데이터별 script | `CommonDataset` + `run_filter()` |
| 상태 | 2D/3D가 실험별로 혼재 | `R,v,p,bg,ba` 관성 15D 구조 |
| 출력 | 저장된 그래프·영상 중심 | estimate CSV + metric JSON + run manifest |
| DR 검증 | local/window drift와 결과 영상 | 초기화 1회의 연속 IMU-only 분리 |
| 학습 결합 | 없음 | CF231 Small TCN velocity + InEKF |

이번 실험은 입력과 설정을 같게 하고 실행 경로를 하나로 만든 것이 구조적인 변경입니다.

## 2. 이번 코드에서 찾은 문제

### 2.1 초기 sample을 한 번 더 적분

이전 `runners/run_filter.py`는 `i=0`에서부터 `predict(control[0], dt[0])`를 호출한 뒤 그 값을 `ground_truth[0]`과 비교했습니다.
필터는 이미 `ground_truth[0]`으로 초기화되어 있어 0번 구간이 이중 반영되었습니다.

수정:

```text
estimate[0] = initial state
for i = 1 ... N-1:
    predict(control[i], dt[i])
```

영향: 첫 시각 오차와 update 횟수가 바로 정렬됩니다. `tests/test_runner_alignment.py`로 회귀 테스트합니다.

### 2.2 InEKF error convention 설명 불일치

코드는 `plus_right(X, delta) = X @ Exp(delta)`를 쓰는데 클래스 설명에는 left correction이 남아 있었습니다. 이 불일치는 나중에 Jacobian이나 update를 바꾸는 사람이 반대 convention을 섞게 만듭니다.

수정:

- error/correction: `X @ Exp(delta)`
- position Jacobian: `H[:, 6:9] = R`
- velocity Jacobian: `H[:, 3:6] = R`
- analytic Jacobian을 central finite difference와 비교
- 실제 update 후 `X` 값이 `plus_right`와 같은지 테스트

코드: `filters/InEKF.py`, `models/Hoon_lie_group_utils.py`, `tests/test_inekf_convention.py`.

### 2.3 sliding-window drift의 과대 해석

이전 검증 중 일부는 1/2/5/10초 window마다 GT `R,v,p`로 다시 초기화했습니다. 이 결과는 짧은 구간의 local drift를 보는 데는 유용하지만 전체 궤적 DR 성능은 아닙니다.

수정: 초기화 1회 후 GT update 0회인 전체 구간을 기본 IMU-only 결과로 사용합니다. window 결과는 local diagnostic으로만 표기합니다.

### 2.4 bias와 GT 사용 범위가 결과표에서 보이지 않음

이전 출력은 최종 RMSE를 보여줘도 초기 bias가 GT인지, 정지 구간 평균인지, 필터가 중간에 바꾸는지를 바로 확인하기 어려웠습니다.

수정: 모든 공통 실행에 `run_manifest.json`을 추가해 resolved bias, gravity, 초기 상태, 데이터 메타데이터, update 횟수를 저장합니다. `metrics.json`에도 bias와 mode를 복제합니다.

### 2.5 Euler 각만으로 자세를 평가

roll/pitch/yaw 차이는 angle wrapping과 pitch 특이점에 민감합니다.

수정: `evaluation/metrics.py`에

```text
R_error = R_gt^T R_est
theta = acos((trace(R_error) - 1) / 2)
```

를 추가해 SO(3) geodesic RMSE와 final error를 함께 저장합니다. 알려진 10° yaw error와 동일 자세에 대한 단위 테스트를 추가했습니다.

### 2.6 재현할 수 없는 과거 수치 혼재

Pohang, i2Nav street01, UrbanNav 과거 표에는 원 설정·loader·output이 없는 값이 포함되어 있었습니다. Pohang은 `baseline.txt`를 update와 evaluation에 같이 쓴 값이어서 독립 위치 성능이 아닙니다.

수정: 해당 CSV·그래프를 공개 결과에서 제거했습니다. `RESULTS.md`는 현재 코드와 설정으로 다시 실행할 수 있는 EuRoC, synthetic motion regime, CF231만 수치를 제시합니다.
