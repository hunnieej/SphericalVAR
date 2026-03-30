# VAR Attention Heatmap 실험 정리

## 목표

Infinity에서 attention heatmap을 `scale`, `layer`, `head` 단위로 추출하여, VAR의 각 scale이 이미지의 `contents`와 `structure`에 어떤 영향을 주는지 분석한다.

핵심 가설은 다음과 같다.

- 어떤 head는 semantic/contextual 정보를 더 잘 유지한다
- 어떤 head는 기하적이거나 반복적인 spatial structure를 더 강하게 표현한다
- 이 성향은 VAR의 coarse-to-fine 생성 구조 때문에 scale에 따라 달라질 수 있다

이 실험은 그 차이를 시각적으로 확인하기 위한 것이다.

## 왜 필요한가

Spherical RoPE를 강하게 적용할수록 다음 현상이 관찰되었다.

- vertical 또는 multi-diagonal attention pattern이 강해짐
- 구조적 연결성은 살아남음
- semantic content는 무너지거나 drift함

즉, 모든 head가 같은 역할을 하지 않고, 모든 scale도 같은 역할을 하지 않을 가능성이 높다. 이를 확인하기 위해 직접 heatmap을 추출한다.

## 실험 아이디어

하나의 inference run에서 self-attention 텐서를 추출하고, 다음 축으로 정리하여 heatmap을 저장한다.

- `scale`
- `layer`
- `head`

이렇게 하면 다음 비교가 가능하다.

- coarse scale vs fine scale
- early layer vs late layer
- structure 중심 head vs content 중심 head

## 모델 세팅

현재 probe 스크립트는 `infinity_2b` 기준으로 구성되어 있다.

- depth: `32` layers
- heads per layer: `16`
- total self-attention heads: `32 x 16 = 512`

즉 하나의 scale stage는 현재 stage token이 `32`개 transformer layer를 모두 통과하는 구조다.

## 현재 heatmap이 의미하는 것

스크립트는 RoPE 적용 이후의 `q`, `k`를 캡처하고, 각 head에 대해 다음 연산을 수행한다.

1. `QK^T` 계산
2. `softmax` 적용
3. 모든 query token에 대해 평균
4. key 축을 다시 `(height, width)`로 reshape

따라서 저장되는 heatmap 한 장은 다음을 의미한다.

- 특정 `layer`
- 특정 `head`
- 특정 `scale`
- 에서 평균적으로 어떤 공간 key 위치가 더 많이 참조되는가

해석:

- 밝을수록 -> 그 위치가 더 자주/강하게 참조됨
- 세로 줄무늬 -> column-wise structural bias 가능성
- 대각선 / multi-diagonal -> structural pattern 가능성
- 좁은 local blob -> local/content-sensitive attention 가능성
- 넓고 diffuse한 분포 -> 전역적 attention 선호

## 중요한 한계

현재 heatmap은 `query-specific` attention map이 아니라, `query-averaged` head summary다.

즉:

- head의 전반적 성향을 보기에는 좋다
- 특정 token 하나가 어디를 보는지 분석하기에는 부족하다

하지만 contextual vs structural head를 1차적으로 구분하는 용도에는 충분히 유용하다.

## 사용 스크립트

현재 heatmap 추출에는 다음 파일을 사용한다.

- `tools/probe_attention_heatmap.py`

지원 기능:

- 조건 선택: `baseline`, `spherical_all`, `spherical_split`
- layer 선택
- 단일 scale / scale 범위 / 모든 scale 선택
- 생성 이미지는 저장하지 않고 heatmap만 저장

## RoPE 조건

### baseline

- 표준 Infinity RoPE
- spherical patch 미적용

### spherical_all

- 모든 head에 spherical-aware RoPE 적용
- `band_ratio`로 width 주파수 대역 중 얼마를 spherical로 바꿀지 제어

### spherical_split

- 일부 head만 spherical-aware RoPE 적용
- `head_split_ratio`로 spherical 그룹에 들어가는 head 비율 제어
- `band_ratio`로 spherical 그룹 내부 width frequency band 중 얼마를 spherical로 바꿀지 제어

현재 구현에서:

- `head_split_ratio`는 `head axis`에 작용
- `band_ratio`는 선택된 spherical head 내부의 `frequency-band axis`에 작용

즉 둘은 중복된 파라미터가 아니다.

## 왜 scale-wise probing이 중요한가

Infinity는 coarse-to-fine 방식의 VAR 모델이다.

각 scale마다 다음 과정이 반복된다.

1. 현재 scale token 준비
2. 모든 `32` layer 통과
3. 현재 scale code 예측
4. 다음 finer scale 생성에 필요한 정보 누적

따라서:

- 같은 layer라도 scale에 따라 다른 역할을 할 수 있음
- coarse scale의 `L31`과 finest scale의 `L31`은 같은 의미가 아님
- content collapse는 잘못된 head를 잘못된 scale에서 건드릴 때 발생할 수 있음

이 때문에 scale을 고정하지 않고 scale-wise로 보는 것이 중요하다.

## 예시 command

### 1. finest scale에서 마지막 4개 layer 보기

```bash
python tools/probe_attention_heatmap.py \
  --condition spherical_split \
  --layers -4 -3 -2 -1 \
  --scale finest \
  --head_split_ratio 0.25 \
  --band_ratio 0.5 \
  --output_dir probe_outputs/attn_heatmaps
```

### 2. 첫 번째 layer를 모든 scale에 대해 보기

```bash
python tools/probe_attention_heatmap.py \
  --condition spherical_split \
  --layers 0 \
  --scales all \
  --head_split_ratio 0.25 \
  --band_ratio 0.5 \
  --output_dir probe_outputs/attn_heatmaps
```

### 3. coarse 첫 scale에서 모든 layer 보기

```bash
python tools/probe_attention_heatmap.py \
  --condition baseline \
  --layers all \
  --scale 0 \
  --output_dir probe_outputs/attn_heatmaps
```

### 4. 일부 layer를 여러 scale 범위에서 보기

```bash
python tools/probe_attention_heatmap.py \
  --condition spherical_split \
  --layers 0 15 31 \
  --scale_range 0 3 \
  --head_split_ratio 0.25 \
  --band_ratio 0.5 \
  --output_dir probe_outputs/attn_heatmaps
```

## 출력 구조

저장 구조 예시는 다음과 같다.

```text
probe_outputs/attn_heatmaps/
  spherical_split/
    L00/
      scale00/
        head00.png
        head01.png
      scale01/
        head00.png
    L31/
      scale12/
        head15.png
```

의미:

- `L00` -> layer 0
- `scale00` -> 첫 번째 VAR scale
- `head15.png` -> 해당 layer/scale의 15번 head

## 우리가 확인하고 싶은 것

이 실험을 통해 다음 질문에 답하고자 한다.

1. 어떤 head가 local/content-preserving pattern을 보이는가?
2. 어떤 head가 vertical stripe나 multi-diagonal 같은 structural pattern을 보이는가?
3. structural pattern은 어느 scale에서 더 강해지는가?
4. spherical RoPE 수정은 어떤 layer/head에서 가장 큰 변화를 일으키는가?
5. content collapse는 잘못된 head 선택 때문인가, 잘못된 scale 선택 때문인가, 아니면 둘 다인가?

## 해석 프레임

### contextual / content-related head로 추정되는 경우

다음과 같은 특성을 기대한다.

- 더 local하거나 semantic-selective한 attention
- rigid한 geometric repetition이 약함
- scene/object consistency 유지에 기여

### structural head로 추정되는 경우

다음과 같은 특성을 기대한다.

- 강한 vertical stripe
- 강한 diagonal / multi-diagonal pattern
- 반복적이거나 기하적인 attention layout

주의:

- 현재 코드에는 head 역할의 ground-truth annotation이 없음
- 따라서 이 분류는 실험적 해석이며, heatmap과 추가 통계 분석을 통해 검증해야 함

## 현재 실험의 의미

이 실험은 Infinity self-attention이

- `scale`
- `layer`
- `head`
- `RoPE condition`

에 따라 어떻게 달라지는지 직접 볼 수 있게 해준다.

즉, aggressive spherical RoPE가 왜 spatial structure는 살리고 semantic content는 망가뜨리는지 진단하기 위한 핵심 도구다.

## 다음 확장 방향

1. center / seam-left / seam-right query에 대한 query-specific heatmap 추가
2. PNG와 함께 raw attention array(`.npy`) 저장
3. 여러 prompt에 대한 variance를 계산해 contextual vs structural head를 자동 분류
4. 동일한 `scale-layer-head`에 대해 baseline vs spherical을 직접 비교
5. 현재 ratio 기반 split 대신, 측정 기반 head split으로 교체
