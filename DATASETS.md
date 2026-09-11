# 데이터 준비

원시 데이터는 저장소에 포함하지 않습니다. 내려받은 데이터는 저장소 루트의 `data/` 아래에 다음과 같이 배치합니다.

```text
data/
├── euroc/
│   └── V1_01_easy/
│       └── mav0/
│           ├── imu0/data.csv
│           └── state_groundtruth_estimate0/data.csv
├── i2nav_robot/
│   └── street00/
│       ├── street00_ADIS16465_IMU.txt
│       ├── street00_F9P_GNSS.pos
│       └── street00_groundtruth.nav
├── pohang_canal/
│   └── pohang05/
│       └── navigation/
│           ├── ahrs.txt
│           └── baseline.txt
└── cf231_leave_one_out/
    └── csv/
        ├── 3/{cf231_imu_raw.csv, poses.csv}
        ├── 4/{cf231_imu_raw.csv, poses.csv}
        ├── 5/{cf231_imu_raw.csv, poses.csv}
        ├── 9/{cf231_imu_raw.csv, poses.csv}
        └── 10/{cf231_imu_raw.csv, poses.csv}
```

## 내려받는 곳

- [EuRoC MAV](https://www.research-collection.ethz.ch/entities/researchdata/bcaf173e-5dac-484b-bc37-faf97a594f1f)
- [Pohang Canal Dataset](https://github.com/dhchung/pohang_canal_dataset)
- [i2Nav-Robot](https://github.com/i2Nav-WHU/i2Nav-Robot)
- [UrbanNav](https://github.com/weisongwen/UrbanNavDataset)

현재 공통 실행기에서 바로 읽을 수 있는 데이터는 EuRoC, i2Nav, Pohang입니다. CF231은 파일 구조와 학습 과정이 달라 `state-estimation-cf231-tcn` 명령으로 따로 실행합니다. UrbanNav loader와 재현 설정은 현재 저장소에 없습니다.

## 데이터별 확인 사항

### EuRoC

`config/euroc.yaml`과 `config/euroc_imu_only.yaml`의 `root`가 `mav0` 폴더를 가리키는지 확인합니다. 현재 설정은 `V1_01_easy`를 사용합니다.

### i2Nav

기본 설정은 `config/i2nav_street00.yaml`입니다. 다른 sequence를 사용하려면 이 파일을 복사한 뒤 `sequence`, `root`, IMU/GT/GNSS 파일명을 함께 바꿉니다. F9P GNSS는 update에, `groundtruth.nav`는 평가에 사용합니다.

### Pohang

현재 loader는 `baseline.txt`에서 위치를 읽어 update와 평가에 모두 사용합니다. 이 상태의 위치 RMSE는 실제 localization 정확도보다 좋게 나올 수 있습니다. 위치 성능을 제대로 비교하려면 RTK-GPS 같은 독립 측정을 update에 넣고 `baseline.txt`는 평가에만 사용해야 합니다.

### CF231

각 run 폴더에 IMU와 pose CSV가 모두 있어야 합니다. `poses.csv`는 학습 run의 속도 정답과 최종 평가에 사용합니다. Run 5의 첫 pose/velocity/orientation과 accelerometer bias의 gravity direction은 GT로 정하고, 그 이후 propagation과 learned update의 runtime sensor는 IMU만 사용합니다.

원시 데이터가 없더라도 `python examples/basic_filter_loop.py`와 합성 회전 실험은 실행할 수 있습니다.
