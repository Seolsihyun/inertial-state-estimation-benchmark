# 데이터 loader 구조

공통 실행기는 설정 파일의 `dataset.type`을 보고 다음 loader를 선택합니다.

| type | loader | 기본 데이터 |
|---|---|---|
| `euroc` | `datasets/euroc.py` | EuRoC MAV CSV |
| `i2nav` | `datasets/i2nav.py` | i2Nav-Robot text/nav 파일 |
| `pohang` | `datasets/pohang.py` | Pohang Canal navigation 파일 |
| `synthetic_motion` | `datasets/synthetic_inertial_dataset.py` | 코드에서 생성한 합성 IMU |

CF231은 여러 run을 학습과 평가로 나누기 때문에 이 dispatcher를 사용하지 않고 `learned/cf231_protocol.py`에서 읽습니다.

## 공통 반환값

각 loader는 `datasets.common.CommonDataset`을 반환합니다. 중요한 field는 다음과 같습니다.

- `timestamps`: IMU timestamp, second
- `controls`: `[ax, ay, az, gx, gy, gz]`
- `dt`: 각 IMU sample 사이의 시간
- `ground_truth`: `[px, py, pz, roll, pitch, yaw]`
- `position_measurements`: 위치 update 값
- `position_measurement_mask`: 각 IMU 시점에 위치 update가 있는지 표시
- `initial_velocity`, `gyro_bias`, `accel_bias`, `gravity`: 필터 초기값과 중력
- `metadata`: frame, 단위, sequence 이름 등

`mode`는 다음 두 값을 사용합니다.

- `imu_only`: IMU predict만 실행
- `fused`: IMU predict와 measurement update 실행

## 새 loader를 추가하는 순서

1. 원본 timestamp와 단위를 SI 단위로 바꿉니다.
2. IMU, update 측정, GT의 시간 범위를 맞춥니다.
3. 센서 좌표계와 world 좌표계를 명확히 정합니다.
4. `CommonDataset`으로 변환합니다.
5. `datasets/loader.py`에 type을 등록합니다.
6. `config/`에 실행 설정을 추가하고 IMU-only와 fused를 각각 확인합니다.
7. update에 사용하지 않은 GT로 위치와 자세 오차를 계산합니다.

파일 배치와 데이터별 주의사항은 상위 폴더의 [DATASETS.md](../DATASETS.md)에 있습니다.
