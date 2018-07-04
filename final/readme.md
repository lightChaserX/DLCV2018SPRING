
## task 1:
sh task1.sh $1=data路徑 $2=輸出路徑 $3=gpu第幾顆 $4=下載model的位置(預設是 ./ )

## task 2:
### one shot
python3 omniglot_test_few_shot_v2.py -w 5 -s 1 -b 15 -e 1 -t 300 -p test -p task2-dataset

### five shot
python3 omniglot_test_few_shot_v2.py -w 5 -s 5 -b 15 -e 1 -t 500 -p test -p task2-dataset

### ten shot
python3 omniglot_test_few_shot_v2.py -w 5 -s 10 -b 15 -e 1 -t 500 -p test -p task2-dataset