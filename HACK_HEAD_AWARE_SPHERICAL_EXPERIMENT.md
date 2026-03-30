# HACK 기반 Head-Aware Spherical RoPE 실험 설계

## 목표
HACK의 attention head 분류를 이용해 Infinity의 spherical 2D RoPE를 head별로 다르게 적용한다.

- Contextual head: 표준 또는 약한 spherical width 보정
- Structural head: 강한 spherical width 보정
- 공통 가설: seam 개선은 structural head가 주도하고, semantic 보존은 contextual head를 덜 건드릴수록 좋아진다.

## 구현 파일
- `head_aware_spherical_rope_infinity.py`
- 기존 참고 구현: `spherical_rope_infinity.py`, `spherical_rope_infinity_v2.py`

## 패처 구조
`HeadAwareSphericalRoPEInfinityPatcher`는 두 개의 RoPE cache를 만든다.

1. contextual grid
- 기본값: `contextual_alpha_w_max=0.0`
- 사실상 표준 RoPE 유지

2. structural grid
- 기본값: `structural_alpha_w_max=1.0`
- fine scale로 갈수록 spherical width 보정 증가

현재 Infinity는 head 공통 cache만 지원하므로, 패처는 `SelfAttention.forward`를 임시 override해서 다음 순서로 동작한다.

1. contextual grid로 q/k rotary 적용
2. structural grid로 q/k rotary 적용
3. head assignment에 따라 structural head만 structural q/k를 사용
4. 나머지 head는 contextual q/k를 사용

## head assignment 파일 형식
다음 셋 중 하나를 쓰면 된다.

### 전역 assignment
```json
{
  "global": {
    "contextual": [0, 1],
    "structural": [2, 3, 4, 5, 6, 7]
  }
}
```

### layer별 assignment
```json
{
  "layers": {
    "0": {"contextual": [0, 1], "structural": [2, 3, 4, 5, 6, 7]},
    "1": {"contextual": [0], "structural": [1, 2, 3, 4, 5, 6, 7]}
  }
}
```

### 바로 쓰는 축약형
```json
{
  "contextual": [0, 1],
  "structural": [2, 3, 4, 5, 6, 7]
}
```

## HACK 분류에서 받아와야 하는 것
오프라인 head classification 결과에서 각 layer의 contextual/structural head index를 JSON으로 저장한다.

최소 필요 정보:
- layer index
- contextual head indices
- structural head indices

HACK 논문 기준 분류 원리:
- low variance head -> contextual
- high variance head -> structural

권장 설정:
- `alpha = 0.3` 근처부터 시작
- 즉 전체 head 중 상위 70% variance를 structural로 우선 가정

## 1차 실험군
### A. baseline
- 원본 Infinity

### B. all-head spherical
- 기존 `spherical_rope_infinity_v2.py`
- 모든 head에 동일한 scale-aware spherical 적용

### C. head-aware v1
- contextual: `alpha_w_max=0.0`
- structural: `alpha_w_max=1.0`
- 추천 시작점

### D. head-aware v2
- contextual: `alpha_w_max=0.2`
- structural: `alpha_w_max=1.0`

### E. head-aware v3
- contextual: `alpha_w_max=0.0`
- structural: `alpha_w_max=0.7`

## 권장 ablation 순서
1. global assignment로 먼저 검증
2. layer-specific assignment로 확장
3. structural alpha 강도 sweep
4. contextual alpha 강도 sweep
5. alpha schedule 비교
- `linear`
- `sigmoid`
- `exp`

## 측정 지표
반드시 같이 기록할 것:
- `seam_ssim`
- `seam_cont`
- `cubemap_clip_equatorial`
- `cubemap_clip_polar`
- 프롬프트 의미 유지 여부에 대한 정성 비교

해석 기준:
- seam 지표만 좋아지고 CLIP이 크게 떨어지면 contextual head를 과도하게 건드린 것
- seam 개선과 CLIP 유지가 동시에 되면 head-aware 설계가 유효

## 사용 예시
```python
from head_aware_spherical_rope_infinity import (
    HeadAwareRoPEConfig,
    HeadAwareSphericalRoPEInfinityPatcher,
)

cfg = HeadAwareRoPEConfig(
    contextual_alpha_w_max=0.0,
    structural_alpha_w_max=1.0,
    alpha_schedule="linear",
    alpha_h=0.0,
    default_contextual_ratio=0.3,
)

patcher = HeadAwareSphericalRoPEInfinityPatcher(
    model=infinity_test,
    assignment_path="probe_outputs/hack_head_assignments.json",
    config=cfg,
)

with patcher:
    img = gen_one_img(...)
```

## 지금 당장 할 일
1. HACK 방식의 offline head classification 결과를 `probe_outputs/hack_head_assignments.json`으로 저장
2. `baseline / all-head spherical / head-aware` 3조건 비교
3. 먼저 125M에서 검증 후 2B로 확대

## 현재 코드의 한계
- 아직 HACK의 attention variance를 직접 계산하는 스크립트는 없다
- assignment JSON이 없으면 기본적으로 앞쪽 일부 head를 contextual로 가정한다
- 따라서 다음 작업은 `offline head classification` 스크립트 추가다
