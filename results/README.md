# 결과 근거 파일

이 폴더는 보고서가 아니라 `RESULTS.md`의 수치를 검산하기 위한 데이터와 그래프를 보관합니다.

| 파일 | 생성 경로 |
|---|---|
| `data/euroc_results.csv` | `runners/run_all.py` + `config/euroc*.yaml` |
| `data/motion_regime_metrics.csv` | `runners/run_motion_regime_benchmark.py` |
| `data/cf231_run5_summary.json` | `runners/run_cf231_small_tcn.py` |
| `data/cf231_learned.csv` | CF231 최종 comparison 3개 method의 metric 정리 |
| `diagnostics/cf231_shallow_learned_velocity.csv` | Earlier diagnostic baseline. Not included in the final reported comparison. |

EuRoC CSV의 SO(3) 지표는 저장된 전체 `estimate.csv`에 현재 `evaluation.metrics.compute_metrics()`를 적용해 계산했습니다. runtime은 장비 상태에 따라 달라지므로 정확도 결론으로 사용하지 않습니다.

새 결과를 추가할 때는 사용한 YAML, commit, `run_manifest.json`, raw output 위치를 함께 기록합니다. 원 설정을 찾을 수 없는 과거 수치는 여기에 넣지 않습니다.
