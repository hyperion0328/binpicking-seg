# selected — 채택한 데이터셋

빈피킹용 소형 부품 탐지/분할에 쓰기로 한 세트. 전부 **근접 촬영 + 부품이 주인공**이고,
소형 구간은 다운스케일로 만든다(원거리 촬영 데이터를 못 찾은 결과 — `../rejected/MANIFEST.md` 참고).

객체 크기는 전부 **외접 축정렬 박스의 `sqrt(area)`** 기준(프로젝트 단일 정의).
"변대비"는 `sqrt(area) / sqrt(W*H)`.

| 폴더 | 이미지 | 인스턴스 | 클래스 | seg 라벨 | 객체 변대비 | 출처 |
|---|---|---|---|---|---|---|
| `rf-screw-nut-bolt` | 1,300 | 27,265 | bolt 9,819 / nut 14,372 / screw 3,074 | **RLE 전량** (마스크/박스 면적비 중앙값 0.538 → 진짜 윤곽) | 5.3% (512×512) | [Roboflow](https://universe.roboflow.com/logesh-s-workspace/screw-nut-bolt) |
| `rf-arg-fixings3` | 274 | 5,557 | Bolt/Nut/Screw/Washer | **불완전** — 폴리곤 1,375 / 박스만 2,188 | 4.3% (1920×1080·4080×3072 혼재) | [Roboflow](https://universe.roboflow.com/bolts/arg_fixings_3) |
| `rf-nuts-and-bolts` | 100 | 579 | Bolt 252 / Nut 256 / washer 71 | **폴리곤 전량** (4점 초과) | 8.4% (640×640) | [Roboflow](https://universe.roboflow.com/jente-kleinh/nuts-and-bolts) |
| `rf-fasteners` | 179 | 538 | bolt 297 / nut 241 | **폴리곤 전량** | 8.6% (640×640) | [Roboflow](https://universe.roboflow.com/obb-8ehdy/fasteners-vbqng) |

합계 **1,853장 / 33,939 인스턴스**, 이 중 seg 마스크 확실한 것 **28,382개**. 전부 CC BY 4.0.

## 쓰기 전에 반드시 처리할 것

- **`id=0` 자리표시자 카테고리를 뺄 것.** 네 세트 모두 COCO `categories`에 인스턴스 0개짜리
  상위 카테고리가 들어 있다(`screw-nut-bolt`, `Bolt-Nut-Screw-Washer`, `nuts-and-bolts` 등,
  `supercategory=none`). 안 빼면 인스턴스 0개 클래스가 낀다 — PCBsmd의 `smd`와 같은 함정.
- **`rf-arg-fixings3`는 seg 학습에 그대로 못 쓴다.** 61%가 박스만 있다. bbox 학습에 쓰거나,
  SAM으로 마스크를 채우려면 먼저 SAM을 검증해야 한다(아래).
- **해상도가 Roboflow 내보내기 기준으로 리사이즈돼 있다.** `screw-nut-bolt`는 512×512
  stretch, `nuts-and-bolts`/`fasteners`는 640×640. 원본 해상도가 아니므로 "원본 기준
  COCO small 비율"을 이 값으로 인용하면 안 된다.

## 미확보

- **Ultralytics `screw-test`** (300장 / 3,387 인스턴스, 톱다운 나사·볼트·너트).
  <https://platform.ultralytics.com/ian-mackenzie/datasets/screw-test> 페이지가 JS 렌더링이라
  다운로드 경로·라이선스·주석 형식을 못 읽었다. 계정이 필요할 수 있음.

## 알려진 한계 — 자연 소형 대조군이 없다

네 세트 모두 객체가 이미지 변의 **1.8~11%** 구간에 있다(p10~p90). 이 프로젝트가 재려는
1% 이하 구간은 어디에도 없어서 다운스케일로 만들어야 하는데, `docs/results.md` 4절이
실측한 대로 **다운스케일 곡선은 진짜 소형 성능의 낙관적 상한**이다. 대조군 없이 곡선만
보고하면 "실제로는 안 통할 걸 통한다고 주장"하게 된다.

원거리 촬영 대조군 후보는 `../rejected/MANIFEST.md`의 "원거리 탐색 결과" 절에 정리해 뒀다.
