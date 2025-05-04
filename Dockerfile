FROM python:3.12-slim

# ensure Python prints immediately (so you’ll see your bootstrap print)
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 1) install system deps (gcc)  
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc \
 && rm -rf /var/lib/apt/lists/*

# 2) copy only requirements.txt, then install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) copy the rest of your code
COPY . .

# 4) unbuffered Python + bootstrap print
CMD ["python","-u","tune.py","--trials","50","--target","0.10"]
