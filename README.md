## 개요

> AI Bootcamp 5인 팀 스터디 그룹의 학습 기록 저장소입니다.

## 🎯 스터디 목표

- GitHub Flow를 활용한 협업 경험 및 지속적인 학습 기록

## 📋 스터디 진행 방식

### 1. 주간 공유 세션

- **일정**: 매주 월요일 오전 10시 ~ 13시
- **첫 세션**: 6월 23일(월)
- **방식**: 각자 학습한 내용을 공유하고 서로 의견을 나누는 쌍방향 소통
    - "이 개념을 이렇게 이해했는데, 어떻게 생각하시나요?"
    - "이 부분이 잘 이해가 안 되는데, 부연 설명해주실 분 계신가요?"

### 2. GitHub을 활용한 학습 기록

- **매일 Issue 생성**: 그날의 학습 계획 (TODO 역할)
- **매일 PR 작성**: 학습한 내용 정리 (TIL 역할)
- **GitHub Flow 적용**: 협업 워크플로우 학습

## 🔄 GitHub Flow 워크플로우

```
1. Issue 생성 → 2. Branch 생성 → 3. 작업 수행 → 4. PR 생성 → 5. 리뷰 & 머지
```

### 세부 진행 방식

1. **Issue 생성**: 학습 계획이나 목표를 Issue로 등록
2. **브랜치 생성**: `feature/이슈번호-날짜-이름` 형식으로 브랜치 생성
    - 예: `feature/3-0623-yuiyeong`
3. **학습 내용 작성**: 개인 폴더에 학습 내용 정리
4. **Pull Request**: 작성한 내용을 PR로 올려 팀원들과 공유
5. **코드 리뷰 & 머지**: 리뷰 완료 후 main 브랜치에 병합

## 📁 디렉토리 구조

```
├── 📂 duhwan/
│   ├── 📂 docs/       # 학습 정리 문서
│   ├── 📂 codes/      # 실습 코드
│   └── 📂 notebooks/  # Jupyter 노트북
├── 📂 juyoung/
│   ├── 📂 docs/
│   ├── 📂 codes/
│   └── 📂 notebooks/
├── 📂 jaeyoon/
│   ├── 📂 docs/
│   ├── 📂 codes/
│   └── 📂 notebooks/
├── 📂 jaehyeong/
│   ├── 📂 docs/
│   ├── 📂 codes/
│   └── 📂 notebooks/
├── 📂 yuiyeong/
│   ├── 📂 docs/
│   ├── 📂 codes/
│   └── 📂 notebooks/
│
└── 📄 README.md
```

## 📝 컨벤션

### Issue 제목 형식

```
[YYYY-MM-DD] 학습 계획
```

### PR 제목 형식

```
[YYYY-MM-DD] TIL
```

### 커밋 메시지 형식

```
prefix: 간단한 설명 (한글)
```

**예시**

- `docs: 머신러닝 기초 개념 정리`
- `feat: 이미지 전처리 함수 구현`
- `fix: 데이터 로딩 오류 수정`

**사용 가능한 prefix**

- `feat`: 기능 개발 관련
- `fix`: 오류 개선 혹은 버그 패치
- `docs`: 문서화 작업
- `test`: test 관련
- `conf`: 환경설정 관련
- `build`: 빌드 작업 관련
- `ci`: Continuous Integration 관련
- `chore`: 패키지 매니저, 스크립트 등
- `style`: 코드 포매팅 관련

## 🚀 시작하기

### 1. Repository 를 로컬에 클론

```
git clone https://github.com/AITeamRemix/growth-journal.git
```

### 2. Dependency 설치

- [poetry](https://python-poetry.org/docs/) 를 사용해서 의존성을 관리합니다.

- 만약, upstage 에서 제공 받은 cloud instance 에서 진행한다면, 아래 명령어를 먼저 실행해주세요.
    ```bash
    bash init-instance.sh
    ```
- 아래의 명령을 실행해서 poetry 및 의존성을 설치합니다.

    ```shell
    pip install -U pip setuptools
    pip install poetry
    poetry install --with dev
    pre-commit install
    ```

### 3. 다음 과정을 반복

1. git switch main
2. **git pull**
3. git branch `feature/[branch name]`
4. git switch `feature/[branch name]`
5. 작업
6. git status
7. git add 작업한 파일
8. **git commit**
    - prefix 를 꼭 달아서 메시지를 적읍시다!
9. **git push** origin `feature/[branch name]`
10. GitHub 에 들어가서 **Pull Request(PR)** 만들기
