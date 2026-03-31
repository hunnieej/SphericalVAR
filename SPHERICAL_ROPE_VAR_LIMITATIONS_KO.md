# Spherical RoPE 적용 방식 정리 및 VAR 관점 한계 분석

## 문서 목적

이 문서는 현재 Infinity 기반 VAR 모델에 적용한 Spherical RoPE 방식이 무엇인지 정리하고, VAR의 coarse-to-fine 생성 특성에서 어떤 한계가 있는지, 특히 `height` 축 보정이 필요한지에 대한 문제의식을 정리한다. 또한 이후 실험 설계 방향도 함께 제안한다.

## 1. 현재 적용한 Spherical RoPE의 핵심 아이디어

현재 구현은 엄밀한 의미의 `3D RoPE`가 아니라, **기존 Infinity의 2D RoPE를 유지하면서 width 축 주파수만 spherical하게 교체하는 방식**이다.

핵심 특징은 다음과 같다.

- 기본 Infinity는 `height`, `width` 두 축에 대해 2D RoPE cache를 미리 계산한다.
- 현재 구현은 이 구조를 유지한 채, `width` 축의 frequency만 spherical frequency로 바꾼다.
- `height` 축은 거의 그대로 유지하며, 기본값은 `alpha_h = 0.0`이다.
- 이 수정은 모든 head / 모든 scale에 일괄 적용되는 것이 아니라, 아래 세 축으로 선택적으로 적용된다.
  - `scale`: 어떤 VAR scale에만 spherical RoPE를 적용할지 (`rope_scales`)
  - `head`: 어떤 head group만 spherical cache를 사용할지 (`head_split_ratio`)
  - `band`: spherical head 내부 width frequency 중 얼마를 spherical로 바꿀지 (`band_ratio`)

즉 현재 방식은 수학적으로는 여전히 2D RoPE이고, 적용 정책만 `scale-head-band` 축으로 제어되는 구조다.

## 2. 일반 2D RoPE와 현재 방식의 차이

### 일반적인 Infinity 2D RoPE

- `height` 위치와 `width` 위치를 기준으로 주파수 grid 생성
- 모든 head가 동일한 RoPE cache 사용
- 모든 scale이 동일한 standard positional basis 사용

### 현재 적용한 Spherical RoPE

- `width` 주파수를 seam-continuity를 고려한 spherical frequency로 교체
- `height`는 대부분 standard 유지 (`alpha_h = 0.0`)
- 일부 head만 spherical cache를 사용하도록 분리 가능
- 일부 scale만 spherical cache를 사용하도록 선택 가능
- 일부 width band만 spherical하게 바꾸도록 제한 가능

정리하면:

- 기존: standard 2D RoPE
- 현재: `width-spherical + head-aware + scale-aware + band-aware` 2D RoPE

## 3. 왜 width만 바꿨는가

현재 구현의 출발점은 panorama의 좌우 seam continuity 문제였다.

- panorama는 좌우 경계가 사실상 원통형으로 연결된다.
- 따라서 `width` 축은 원형 위상 조건을 만족하도록 만들 필요가 있다.
- 반면 `height` 축은 기본적으로 그런 주기 조건을 갖지 않기 때문에, 초기 설계에서는 `height`를 standard로 두었다.

즉 현재 기본 가정은 다음과 같다.

- seam 문제의 본질은 `width`
- 따라서 `width`만 spherical하게 바꾸면 충분할 수 있다

하지만 이 가정은 **VAR 구조**에서는 다시 검토가 필요하다.

## 4. VAR 관점에서 현재 방식의 한계

Diffusion 모델과 달리 Infinity는 고정 해상도 latent에서 반복 denoising하지 않는다. 대신 scale schedule을 따라 coarse-to-fine 방식으로 token grid 자체가 계속 바뀐다.

이 차이 때문에 다음 문제가 생긴다.

### 4.1 scale마다 `(ph, pw)`가 달라진다

- coarse scale과 fine scale은 서로 다른 spatial grid를 가진다
- 동일한 layer라도 scale에 따라 positional basis의 의미가 달라질 수 있다

### 4.2 KV cache가 scale을 넘어서 누적된다

- VAR에서는 이전 scale의 key/value가 이후 scale attention에 계속 남는다
- 따라서 positional basis가 scale 사이에서 충분히 일관적이지 않으면, coarse scale semantic scaffold와 fine scale detail stage 사이의 mismatch가 누적될 수 있다

### 4.3 width만 spherical하게 바꾸면 height 축은 그대로 남는다

- 현재는 seam continuity를 해결하기 위해 width만 적극적으로 보정한다
- 하지만 VAR에서는 `height`도 scale에 따라 grid meaning이 바뀌므로, vertical consistency 문제가 남을 수 있다
- 특히 coarse-to-fine semantic 전달에서 `height` 축이 standard 그대로인 것이 안정적인지 재검토가 필요하다

즉 panorama seam 문제 관점에서는 width-only spherical이 타당하지만, VAR의 multi-scale positional consistency 관점에서는 불완전할 수 있다.

## 5. 현재 관찰된 실험적 문제와 연결

지금까지의 관찰을 정리하면 다음과 같다.

- spherical RoPE를 강하게 적용할수록 vertical / multi-diagonal pattern이 강해짐
- structure는 살아남지만 contents generation이 무너지는 경우가 있음
- `head_split_ratio`, `band_ratio`, `rope_scales` 조합에 따라 결과가 달라지지만, `width` 중심 보정만으로는 contextual head의 semantic binding을 충분히 보존하지 못할 가능성이 있음

이 문제는 단순히 head split의 문제만이 아니라,

- `height` 축을 전혀 조정하지 않는 점
- VAR scale 사이 positional basis mismatch 가능성

과도도 연결될 수 있다.

## 6. height를 그대로 두어도 되는가?

현재 시점의 답은 다음과 같다.

### panorama seam 문제만 보면

- `height`를 그대로 두는 것이 자연스럽다
- seam은 본질적으로 width의 주기성과 직접 연결되기 때문

### VAR 전체 positional consistency를 보면

- `height`도 아무 조정 없이 그대로 두는 것이 최선이라고 단정하기 어렵다
- 특히 scale이 변할 때 vertical 방향 의미가 얼마나 안정적으로 유지되는지 점검이 필요하다

즉,

- `height`를 width처럼 완전히 spherical하게 만들 필요가 있는지는 아직 불분명
- 하지만 `height`에도 어떤 형태의 scale-aware correction이 필요할 가능성은 충분하다

## 7. 가능한 height 보정 방향

현재 기준으로 생각할 수 있는 방향은 크게 세 가지다.

### 방향 A: 현 상태 유지

- `alpha_h = 0.0`
- width만 spherical하게 보정

장점:

- 가장 보수적
- seam 개선이라는 목표에 가장 직접적

단점:

- VAR scale consistency 측면에서 height 축이 완전히 방치될 수 있음

### 방향 B: 약한 height spherical 보정

- `alpha_h`를 0보다 조금 크게 설정
- 예: `0.05`, `0.1`, `0.2`

장점:

- vertical consistency를 일부 보완할 수 있음

단점:

- semantic drift나 unwanted structural bias가 생길 수 있음
- width와 같은 강한 spherical 처리는 과할 수 있음

### 방향 C: height scale-aware correction

- `height`는 spherical로 직접 바꾸기보다 scale-aware normalization 또는 adaptive frequency rescaling을 적용
- 즉 pole/seam 목적이 아니라 VAR scale consistency 목적의 보정

장점:

- VAR 구조에 더 직접적으로 대응

단점:

- 구현 복잡도 증가
- 실험 설계가 더 필요함

현재 추천은 **방향 B와 C를 단계적으로 확인하는 것**이다.

## 8. 실험 설계 제안

### 8.1 1차 실험: alpha_h ablation

목적:

- height 축을 아주 약하게 보정했을 때 content/structure tradeoff가 어떻게 변하는지 확인

권장 설정:

- `alpha_h = 0.0`
- `alpha_h = 0.05`
- `alpha_h = 0.1`
- `alpha_h = 0.2`

고정 추천:

- `rope_scales = finest`
- `condition = spherical_split` 또는 `spherical_all`
- `head_split_ratio`, `band_ratio`는 이미 잘 관측되는 조합 하나를 고정

### 8.2 2차 실험: rope_scales와 alpha_h의 상호작용

목적:

- height 보정이 coarse scale에 더 민감한지, finest scale에 더 민감한지 확인

권장 비교:

- `rope_scales = finest`
- `rope_scales = all`
- `rope_scales = 0 1 2`

각각에 대해 `alpha_h`를 바꿔 비교

### 8.3 3차 실험: head-aware 유지 여부 검토

목적:

- height 보정이 contextual head를 얼마나 손상시키는지 확인

권장 비교:

- `spherical_all`
- `spherical_split`

이때 query-centered relative map을 같이 보면,

- local/contextual head가 더 local하게 남는지
- structural head의 directional pattern이 어떻게 변하는지

를 더 잘 볼 수 있다.

## 9. 분석 시 같이 봐야 할 지표

height 보정 실험에서는 단순 생성 이미지뿐 아니라 아래를 함께 보는 것이 좋다.

- generated image quality
- seam continuity
- query variance map
- query-centered relative map
- layer/head별 contextual vs structural 패턴 변화

특히 relative map은,

- query 기준으로 어떤 방향의 key를 선호하는지
- local blob / directional band / symmetric pattern이 어떻게 달라지는지

를 볼 수 있어서 height 보정 효과를 분석하는 데 유용하다.

## 10. 현재 결론

현재 width-only spherical RoPE는 panorama seam 문제에는 자연스러운 출발점이지만, VAR의 scale-varying 구조를 고려하면 충분조건이라고 보기 어렵다.

핵심 문제의식은 다음과 같다.

- seam continuity는 width가 핵심이다
- 하지만 VAR의 multi-scale positional consistency는 width만의 문제가 아닐 수 있다
- 따라서 `height`에도 최소한 약한 보정 또는 scale-aware correction이 필요할 가능성이 있다

즉 현재 구현은 좋은 1차 베이스라인이지만, VAR에 truly adapted된 positional design이라고 보기는 어렵다.

## 11. 추천 다음 단계

1. `alpha_h` ablation 추가
2. `rope_scales`별로 alpha_h 영향 비교
3. `relative map` 기준으로 directional/locality 변화 분석
4. content collapse와 seam improvement를 함께 평가
5. 필요하면 height를 spherical이 아니라 scale-aware normalization 방향으로 재설계

---

요약하면, 현재 구현은 **width seam 문제를 해결하기 위한 sphericalized 2D RoPE**이며, VAR 전체 구조를 고려하면 `height` 축도 추가 보정 후보로 보는 것이 합리적이다.

## 12. Relative Map 의미
 relative layer map은 정확히는 query-centered relative map이고, 의미는 다음과 같습니다.
- 일반 attn_mean 맵
  - “이미지의 절대 위치 중 어디를 많이 보느냐”
- relative 맵
  - “각 query를 중심으로 봤을 때, 상대적으로 어느 방향/거리의 key를 많이 보느냐”
즉 절대 좌표계를 버리고, 모든 query를 자기 자신 중심으로 정렬해서 평균낸 맵이에요.
직관
예를 들어 어떤 query가 있다고 할 때:
- 오른쪽 이웃을 자주 본다
- 위쪽 먼 곳을 자주 본다
- 자기 자신 주변만 본다
이런 패턴을 query 위치와 무관하게 모아서 보는 겁니다.
그래서 relative map에서:
- 중심
  - query 자기 자신 위치에 대응
- 중심 주변이 밝음
  - local attention
  - contextual/local head 가능성
- 중심 오른쪽만 밝음
  - query 기준 오른쪽 방향을 선호
- 중심 위/아래 띠가 밝음
  - 특정 방향성 구조를 선호
- 대각 방향이 밝음
  - diagonal/multi-diagonal structural pattern 가능성
- 멀리 떨어진 위치가 대칭적으로 밝음
  - 반복 구조, 장거리 관계를 보는 head일 수 있음
왜 absolute map이랑 다르냐
- absolute map은 “화면의 어디를 보는가”라서 scene content나 위치 편향이 섞입니다
- relative map은 “query 기준으로 어떤 관계를 보는가”라서
  - head의 attention kernel 성질
  - local vs directional vs periodic 구조
를 더 잘 드러냅니다
현재 구현에서의 의미
지금 relative map은:
- final scale 내부 query/key만 대상으로
- 각 query-key attention을
- (key_h - query_h, key_w - query_w) 상대 좌표로 옮겨 담고
- 모든 query에 대해 평균낸 결과입니다
그래서 이 map은 질문에 답합니다:
- 이 head는 보통 query 주변을 보는가?
- 특정 방향을 보는가?
- 반복적인 구조를 보는가?
- local content binding을 하는가?
해석 팁
- attn_mean과 같이 보면
  - 절대적으로 어디를 보는지
- relative와 같이 보면
  - 관계적으로 어떻게 보는지
를 동시에 볼 수 있습니다
한 줄 요약:
- relative map은 “이 head가 query를 중심으로 봤을 때 어떤 상대 위치의 token을 선호하는가”를 보여주는 맵입니다.

