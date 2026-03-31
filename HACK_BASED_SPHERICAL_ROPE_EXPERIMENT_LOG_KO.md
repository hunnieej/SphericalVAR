# HACK 기반 Spherical RoPE 실험 정리

## 문서 목적

이 문서는 Infinity-2B 기반 panorama 생성 실험에서 Spherical RoPE를 적용하게 된 배경, HACK(Head-aware KV Cache Compression) 스타일 분석을 도입한 이유, 실제로 진행한 실험들, 그리고 현재 코드 구현 상태까지의 흐름을 한국어로 정리한 기록이다.

## 1. 출발점: 왜 Spherical RoPE를 적용하려 했는가

초기 문제의식은 panorama 이미지 생성에서 좌우 seam continuity를 개선하는 것이었다.

- 기본 Infinity는 2D RoPE를 사용한다.
- panorama는 좌우가 원통형으로 연결되기 때문에, width 축의 positional phase가 seam에서 자연스럽게 이어지는 것이 중요하다.
- 이를 위해 width 축 frequency를 spherical하게 보정하는 Spherical RoPE를 Infinity에 주입하는 방향을 실험하기 시작했다.

초기 구현의 핵심은 다음과 같았다.

- 기본 Infinity의 2D RoPE 구조는 유지
- width 축 frequency만 spherical frequency로 교체
- height 축은 기본적으로 standard 유지 (`alpha_h = 0.0`)

## 2. VAR 구조를 고려해야 한다는 문제의식

실험을 진행하면서 단순히 seam 문제만이 아니라, Infinity가 VAR(coarse-to-fine autoregressive) 구조라는 점이 중요하다는 것을 확인했다.

핵심 관찰:

- Diffusion 모델처럼 고정 해상도 latent에서 반복 denoising하는 구조가 아님
- scale schedule을 따라 token grid 자체가 변함
- coarse scale의 key/value가 이후 fine scale attention에 KV cache로 누적됨
- 따라서 positional basis의 작은 변화도 scale 전체로 전파될 수 있음

이 때문에 다음과 같은 질문이 생겼다.

- Spherical RoPE를 모든 scale에 동일하게 넣는 것이 맞는가?
- 모든 head에 동일하게 넣는 것이 맞는가?
- width만 보정하고 height는 전혀 건드리지 않아도 되는가?

## 3. Flash / Flex 없이 fallback 경로 정리

실험 초기에 Ubuntu / flash-attn 환경 이슈로 인해, 모델이 flash attention import 단계에서 죽는 문제가 있었다.

이를 해결하기 위해:

- `flash_attn` import를 안전하게 fallback 처리
- `SelfAttention`, `CrossAttention` 모두 slow path로 동작 가능하게 수정
- `tools/run_infinity.py --use_flex_attn 0`로 flash/flex 없이도 실행 가능하게 정리

이 단계는 이후 모든 RoPE 실험을 안정적으로 진행하기 위한 기반 작업이었다.

## 4. Head split / band split 실험으로 확장

초기 Spherical RoPE는 사실상 모든 head에 width spherical 보정을 거는 실험에 가까웠다.

그러나 실험 결과:

- structure는 살아나지만
- contents generation이 쉽게 무너지는 현상

이 관찰되었다.

그래서 다음 두 축을 도입했다.

### 4.1 head split

- 일부 head만 spherical cache를 사용
- 나머지는 standard RoPE 유지
- 초기 방식은 ratio 기반 분할 (`head_split_ratio`)

예:

- `0.25`면 일부 head만 spherical
- `0.75`면 더 많은 head가 spherical

### 4.2 band split

- spherical head 내부에서도 width frequency 전체를 바꾸지 않고
- 일부 high-frequency band만 spherical로 교체
- `band_ratio`로 제어

이 단계에서 중요한 결론:

- `head_split_ratio`는 head 축에서 작동
- `band_ratio`는 head 내부 frequency band 축에서 작동
- 둘은 다른 차원의 제어 파라미터임

## 5. scale-aware 적용으로 확장

이후 Spherical RoPE를 특정 scale에만 적용할 수 있도록 확장했다.

도입된 파라미터:

- `rope_scales`

예:

- `coarsest` = 첫 번째 scale만
- `finest` = 마지막 scale만
- `0 1 2` = 특정 초기 scale만
- `all` = 모든 scale

이 확장은 매우 중요했다. 왜냐하면:

- coarse scale은 semantic scaffold를 만들고
- fine scale은 detail / seam 보정에 더 민감하기 때문

즉, “어느 scale에 spherical RoPE를 넣는가”가 결과를 크게 바꿀 수 있다는 가설을 실험할 수 있게 되었다.

## 6. HACK 스타일 분석이 필요해진 계기

실험을 진행하면서 중요한 문제가 생겼다.

- head split ratio와 band ratio를 바꿔도 어떤 경우에는 생성 이미지 차이가 작게 보이거나
- 어떤 head가 실제로 contextual 역할인지, structural 역할인지 불명확했다

이때 HACK(Head-aware KV Cache Compression for Efficient Visual AutoRegressive Modeling)의 head classification 아이디어를 참고하게 되었다.

핵심 포인트:

- 단순히 한 장의 attention map을 눈으로 보고 head를 분류하는 것이 아님
- final generation step의 attention weight를 수치화해서 분류
- query 방향 분산이 낮으면 contextual
- query 방향 분산이 높으면 structural

이 접근은 현재 실험과 매우 잘 맞았다.

이유:

- Infinity도 VAR 구조
- final step은 이전 모든 scale의 상호작용이 통합되는 시점
- head 역할을 정량적으로 나눌 필요가 있었음

## 7. HACK 기반 head classification 구현

이를 위해 `tools/probe_head_classification.py`를 새로 구축/확장했다.

현재 이 스크립트는:

- actual attention weight 기반으로 capture
- final generation step만 사용
- layer all 기준으로 모든 layer-head를 분석
- prompt 여러 개를 돌려 query variance score 계산
- global quantile threshold로 contextual / structural 분류

출력:

- `json` 분류 결과
- `attn_mean` heatmap
- `query_var` heatmap

이 단계에서 단순한 q/k proxy heatmap이 아니라, 실제 final-step attention 기반 분석으로 넘어갔다.

## 8. Query-centered relative map 도입

absolute attention map만으로는 “이미지 어디를 보느냐”는 볼 수 있지만,

- 각 query를 중심으로 어떤 상대 방향/거리의 key를 선호하는지
- structural head가 directional pattern을 갖는지
- contextual head가 local blob 형태를 갖는지

를 보기 어렵다는 문제가 있었다.

그래서 `query-centered relative map`을 추가했다.

핵심 아이디어:

- final-scale attention에서 current-scale query/key만 사용
- 각 query-key attention을 절대 위치가 아니라 `(key - query)` 상대 좌표로 재배치
- 모든 query에 대해 평균

이로 인해:

- local/contextual head
- directional/structural head
- periodic/multi-diagonal head

를 더 구조적으로 분석할 수 있게 되었다.

## 9. Heatmap / generated image 시각화 체계 정리

실험이 많아지면서 heatmap과 생성 이미지를 함께 보는 구조도 정리했다.

현재는:

- scene별 generated image 저장
- `attn_mean`, `query_var`, `relative` 저장
- 같은 head를 모든 layer에 대해 모은 4x8 grid 생성
- 같은 layer 안에서 모든 head를 모은 4x4 grid 생성
- 각 grid 위에 생성 이미지를 붙인 composite 이미지 생성

즉 현재 scene별로 다음 비교가 가능하다.

- 같은 head가 layer에 따라 어떻게 변하는가
- 같은 layer 안에서 어떤 head가 contextual / structural처럼 보이는가
- 생성 이미지와 attention 패턴을 함께 비교

## 10. Sweep 스크립트와 대규모 실험 자동화

실험 조합이 많아지면서 자동화 스크립트도 구축했다.

### 10.1 run_comparison_2b 기반 실험

- `run_comparison_2b.py`를 중심으로
  - `head_split_ratio`
  - `band_ratio`
  - `rope_scales`
  - `interp_mode`
를 바꾸는 실험을 진행

### 10.2 probe_head_classification sweep

- `tools/run_probe_head_classification_sweep.sh`
- baseline / spherical_all / spherical_split를 포함
- 이미 끝난 실험 skip
- progress 표시
- log는 파일로 저장

이로 인해 대규모 조합 실험과 재시작이 가능해졌다.

## 11. Height에 대한 새로운 문제의식

초기 Spherical RoPE는 seam 문제 때문에 width 중심으로 설계되었다.

그러나 VAR 구조를 고려하면 다음 의문이 생겼다.

- width만 spherical하게 바꾸고 height는 standard로 유지해도 충분한가?
- scale에 따라 grid가 달라지는데, height도 scale-aware correction이 필요한 것 아닌가?

현재 코드상:

- `alpha_h`는 이미 patcher 내부 수식에 존재
- 하지만 주요 실행 스크립트에서는 전부 `alpha_h=0.0`으로 고정

즉 수학적 지원은 있으나, 실험 파라미터로 아직 본격적으로 열어두진 않은 상태다.

이 문제의식은 별도 문서 `SPHERICAL_ROPE_VAR_LIMITATIONS_KO.md`에도 정리되어 있다.

## 12. 현재 코드 구현 상태 요약

현재 코드에서 가능한 것:

- flash/flex 없이 slow path inference 가능
- width spherical frequency 적용
- `head_split_ratio` 기반 head split
- `band_ratio` 기반 width frequency band 제한
- `rope_scales` 기반 scale selective 적용
- HACK 스타일 final-step head classification
- `attn_mean`, `query_var`, `relative` heatmap 저장
- generated image + grid composite 저장

추가로 최근 확장된 기능:

- 특정 layer, 특정 head만 spherical RoPE를 적용하는 exact selection
- 예: `0:5,6;6:6`

즉 이제는 ratio 기반 실험뿐 아니라,

- 첫 번째 layer의 5,6번째 head
- 여섯 번째 layer의 6번째 head

같은 exact head-level intervention 실험까지 가능해졌다.

## 13. 현재까지의 핵심 교훈

지금까지의 실험을 통해 확인한 핵심은 다음과 같다.

1. Spherical RoPE는 seam continuity 개선에 유망하다.
2. 그러나 모든 scale / 모든 head / 모든 band에 강하게 넣으면 contents generation이 쉽게 무너진다.
3. VAR에서는 scale-aware, head-aware 제어가 매우 중요하다.
4. head의 역할은 실제 attention 통계와 relative map을 통해 봐야 한다.
5. width-only 설계는 좋은 출발점이지만, height 보정 가능성도 열어두어야 한다.

## 14. 앞으로의 직접적인 목표

현재 최종 목표는 다음과 같다.

- Infinity-2B 세팅에서
- spherical RoPE를 적용할 `layer`와 `head`를 정확히 지정하고
- HACK 기반 분석으로 contextual / structural head를 식별한 뒤
- 해당 head들에만 selective하게 spherical RoPE를 넣는 실험을 진행하는 것

즉 앞으로의 방향은:

- ratio 기반 대략적 split
- -> HACK 기반 정량 분류
- -> exact layer-head 지정 주입

으로 정리할 수 있다.

## 15. 정리

이번 빌드업의 흐름은 다음 한 줄로 요약할 수 있다.

> panorama seam 개선을 위해 시작한 width-spherical RoPE 실험이, VAR의 multi-scale 구조와 head 역할 분석 문제를 만나면서, HACK 기반 head classification과 query-centered relative map까지 포함하는 정교한 head-aware / scale-aware 실험 체계로 확장되었다.
