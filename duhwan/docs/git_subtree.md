# 🔗 GitHub 저장소 통합 가이드

## 📋 현재 상황 분석

| 항목 | 내용 |
|------|------|
| **타겟 저장소** | `https://github.com/AITeamRemix/growth-journal` |
| **소스 저장소** | `https://github.com/korea202/gx-train.git` |
| **목표** | `growth-journal`의 `duhwan/codes` 디렉토리 밑에 `gx-train` 내용 추가 |

## 🚀 단계별 실행 방법

### 1️⃣ growth-journal 저장소 클론

```bash
git clone https://github.com/AITeamRemix/growth-journal.git
cd growth-journal
```

### 2️⃣ Git Subtree를 사용하여 gx-train 추가

```bash
git subtree add --prefix=duhwan/codes/gx-train https://github.com/korea202/gx-train.git main --squash
```

### 3️⃣ 변경사항 푸시

```bash
git push origin main
```

## 🔄 향후 업데이트 방법

원본 저장소(`gx-train`)에 새로운 변경사항이 있을 때마다 다음 명령어를 실행하세요:

```bash
git subtree pull --prefix=duhwan/codes/gx-train https://github.com/korea202/gx-train.git main --squash
```

## ⚠️ 주의사항

### 사전 확인 필요 사항

- [ ] `duhwan/codes` 디렉토리가 이미 존재하는지 확인
- [ ] 기존 파일들과 충돌이 없는지 확인
- [ ] 저장소 소유자의 허가 필요 여부 확인

### 예상 디렉토리 구조

통합 후 예상되는 디렉토리 구조:

```
growth-journal/
├── 📂 duhwan/
│   ├── 📂 docs/
│   ├── 📂 codes/
│   │   └── 📂 gx-train/    # ← 새로 추가될 내용
│   └── 📂 notebooks/
├── 📂 juyoung/
├── 📂 jaeyoon/
├── 📂 jaehyeong/
├── 📂 yuiyeong/
└── 📄 README.md
```
