#!/bin/bash

set -e  # 에러 발생시 스크립트 중단

echo "시스템 패키지 업데이트 중..."
apt update
apt upgrade -y

echo "Python 빌드에 필요한 의존성 패키지 설치 중..."
apt install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
libffi-dev liblzma-dev git

echo "pyenv 설치 중..."
curl -fsSL https://pyenv.run | bash

echo "환경변수 설정 중..."
# .bashrc에 pyenv 설정 추가
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> /root/.bashrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> /root/.bashrc
echo 'eval "$(pyenv init -)"' >> /root/.bashrc

# .profile에도 추가 (로그인 시 로드됨)
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> /root/.profile
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> /root/.profile
echo 'eval "$(pyenv init -)"' >> /root/.profile

echo "현재 세션에 pyenv 환경변수 적용 중..."
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

echo "Python 3.11.11 설치 중... (시간이 오래 걸릴 수 있습니다)"
pyenv install 3.11.11

echo "Python 3.11.11을 전역 기본 버전으로 설정 중..."
pyenv global 3.11.11

echo "설치 완료!"
echo "Python 버전 확인:"
python --version
python -c "import ssl, bz2, curses, ctypes, readline; print('모든 필수 모듈이 정상적으로 로드되었습니다.')"

echo "환경 다시 로드"
source /root/.bashrc

echo "poetry 설치"
pip install -U pip setuptools
pip install poetry

echo "poetry 를 사용해서 의존성 설치"
poetry install --with dev

echo "코드 스타일 통일화를 위해 pre-commit 설정"
pre-commit install
