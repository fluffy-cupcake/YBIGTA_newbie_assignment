
# anaconda(또는 miniconda)가 존재하지 않을 경우 설치해주세요!
## TODO
if ! command -v conda &> /dev/null
then
    echo "[INFO] Conda가 설치되어 있지 않아 Miniconda를 설치합니다."

    # Miniconda 설치 스크립트 다운로드
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh

    # 설치 스크립트 실행
    bash miniconda.sh -b -p $HOME/miniconda

    # Conda 초기화
    eval "$($HOME/miniconda/bin/conda shell.bash hook)"

    conda init

    echo "[INFO] Miniconda 설치 완료!"
else
    echo "[INFO] Conda가 이미 설치되어 있습니다."
    eval "$(conda shell.bash hook)"
fi


# Conda 환셩 생성 및 활성화
## TODO
conda create --name myenv python==3.11 -y
conda activate myenv 

## 건드리지 마세요! ##
python_env=$(python -c "import sys; print(sys.prefix)")
if [[ "$python_env" == *"/envs/myenv"* ]]; then
    echo "[INFO] 가상환경 활성화: 성공"
else
    echo "[INFO] 가상환경 활성화: 실패"
    exit 1 
fi

# 필요한 패키지 설치
## TODO
pip install mypy

# Submission 폴더 파일 실행
cd submission || { echo "[INFO] submission 디렉토리로 이동 실패"; exit 1; }

for file in *.py; do
    problem_num="${file%.*}"
    input_file="../input/${problem_num}_input"
    output_file="../output/${problem_num}_output"

    echo "[INFO] 실행 중: $file"
    python "$file" < "$input_file" > "$output_file"

done

# mypy 테스트 실행 및 mypy_log.txt 저장
## TODO
mypy . > ../mypy_log.txt

# conda.yml 파일 생성
## TODO
conda env export > ../conda.yml

# 가상환경 비활성화
## TODO
conda deactivate