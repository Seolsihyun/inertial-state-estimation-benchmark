# 코드 수정 시 확인할 사항

## 공통 규칙

1. IMU 입력 순서는 `[ax, ay, az, gx, gy, gz]`로 유지합니다.
2. pose 출력은 `[px, py, pz, roll, pitch, yaw]`이며 SI 단위와 radian을 사용합니다.
3. ground truth를 초기화·calibration·training·update·evaluation 중 어디에 사용했는지 각각 기록합니다.
4. 결과를 추가할 때는 이번 코드로 직접 실행한 값인지, 과거 자료에서 가져온 값인지 적습니다.
5. 원시 데이터, 개인 PC의 절대경로, model checkpoint와 `outputs/`는 commit하지 않습니다. 공개 수치에는 개인 경로를 제거한 manifest와 설정을 함께 남깁니다.

## 수정 후 확인

```bash
pip install -e ".[test,learned]"
python -m pytest -q
python examples/basic_filter_loop.py
```

결과 수치가 달라지는 수정은 사용한 설정 파일, 데이터 sequence, random seed, 위치 update 주기와 trajectory 정렬 방법을 함께 기록합니다. 관련 CSV나 JSON과 그림도 같은 실행 결과로 다시 만듭니다.

## 데이터 loader 추가

새 loader는 `datasets.common.CommonDataset`을 반환해야 합니다. 다음 내용을 코드와 문서에 함께 적습니다.

- 센서 frame과 world frame 정의
- timestamp 단위와 IMU·측정·GT를 맞추는 방법
- IMU 단위 변환
- 초기 위치, 속도, 자세를 정하는 값
- 필터 update에 넣는 측정과 주기
- 마지막 오차 계산에 사용하는 독립적인 기준값

하나의 위치 trajectory를 filter update와 평가에 동시에 사용하면 위치 오차가 실제보다 작아집니다. 부득이하게 사용한 경우에는 결과표와 그래프에 그 사실을 표시합니다.
