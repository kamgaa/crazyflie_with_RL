# Crazyflie 기체 모델

- PhysX asset: `src/crazyflie_isaaclab/assets/cf21b.usda`
- MuJoCo 모델: `resources/mujoco/cf21B_500.xml`
- 원래 mesh: `resources/mujoco/assets/cf21B/*.stl`
- 원본 Git 기준: `648b2010ac3b8d8e784cf2f437de02dc30bf53df`

USD의 시각 형상은 도식적인 기체이며 STL 외형을 자동 변환한 것이 아닙니다.
PhysX에서는 단일 강체로 사용합니다. 자식 질량을 포함한 총질량, COM 및 locked-child
합성 관성을 적용하고 자유 propeller hinge 동역학은 생략합니다.
소프트웨어 모터 RPM 상태는 별도로 유지합니다. 기본 Isaac Crazyflie asset 물성을 사용하지 않습니다.

본체 질량 0.04338kg, 총질량 0.043384kg.
COM (body FLU, m): [0, 0, 1.1063986724568764e-6].
관성 대각 (kg m²): [2.3960577265292863e-5, 2.3960577265292863e-5, 3.236100263679999e-5].
Native 로터 XY (m): [(+.03536,-.03536),(-.03536,-.03536),(-.03536,+.03536),(+.03536,+.03536)].
Native 반토크 부호 [-,+,-,+], user_frd 1~4 -> native [3,2,1,0].
내부 world/body 좌표계는 user_frd 명명 때문에 FRD로 바뀌지 않습니다.

`build_asset.py`는 같은 상수를 읽어 USD를 재생성합니다. 일반 재생에는 필요하지 않습니다.
STL/XML은 원래 저장소의 자산이며 새로운 라이선스를 임의로 부여하지 않습니다.
