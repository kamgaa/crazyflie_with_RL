# Runtime resources

이 디렉터리는 실행에 필요한 정적 리소스의 공통 루트다. 저장소의 Python 코드에는 외부 컴퓨터의 절대경로를 넣지 않고, `configs/base.yaml`의 `paths`만 사용한다.

## Required layout

```text
resources/
├── mujoco/
│   ├── cf21B_500.xml
│   ├── meshes/
│   └── textures/
└── pretrained_models/
    ├── residual/
    └── e2e/
```

현재 저장소에는 `cf21B_500.xml`과 그 종속 mesh·texture가 없다. XML을 임의로 만들거나 종속 파일의 경로를 추측하지 않는다. 원본 XML을 확보한 뒤 `resources/mujoco/cf21B_500.xml`에 두고, XML이 참조하는 모든 파일을 XML의 상대경로와 정확히 일치하도록 `meshes/`, `textures/` 등에 배치해야 한다.

기존 `model/`의 체크포인트는 자동으로 이 디렉터리로 옮기지 않는다. 승인 전 이전 후보와 해시는 [`LEGACY_MIGRATION_INVENTORY.md`](../LEGACY_MIGRATION_INVENTORY.md)에 기록되어 있다.

## Validation after placement

리소스를 배치한 환경에서 다음 명령으로 XML과 모든 상대 참조를 실제 MuJoCo 로더로 검증한다.

```bash
python -c "from pathlib import Path; import mujoco; p=Path('resources/mujoco/cf21B_500.xml').resolve(); m=mujoco.MjModel.from_xml_path(str(p)); print(f'loaded: {p} (nbody={m.nbody})')"
```

파일이 없거나 mesh·texture 참조가 깨졌다면 학습을 시작하지 말고 해당 누락 경로를 수정해야 한다.
