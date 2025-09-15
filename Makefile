# modelのresultsとlogsディレクトリの内容を削除
clean:
	(cd ./model; \
	rm -rf results/*; \
	rm -rf logs/*)

# detasetsの各データセットのロードを実行
amazon:
	(cd ./datasets/codes; \
	python3 load_amazon.py)

google_play:
	(cd ./datasets/codes; \
	python3 load_google_play.py)

datasets: amazon google_play

# modelの学習を実行
train:
	(cd ./model; \
	python3 train.py)

# modelの学習をデータセットのロードから実行
load_train: datasets train

# アプリケーションの実行
run:
	(cd ./app; \
	python3 main.py)

# データのロードとモデルの学習、アプリケーションの実行を一括で実行
all: load_train run

