## resnet50 eager测试
测试eager模式下oneflow与pytorch训练resnet50的效率对比;

### 1. 准备测试环境
* 这里我们在pytorch docker中进行测试，需要先卸载pytorch，再安装cuda11.8对应的版本;
* `pip uninstall -y torch`
* `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
### 2. 克隆仓库
```bash
git clone https://github.com/Oneflow-Inc/models.git
```

### 3. 路径修改
将args_test_speed.sh中的/path/to/修改为自己的绝对路径
有model与logdata两个地方;

### 4. 运行

- 运行之前需要先确认

    - 4个环境变量——在resnet50_test.sh中修改  默认都为true
    - 测试用例（batch size & ddp卡数）——在resnet50_test.sh中修改
      - world_size表示ddp卡数, input shape的第一个表示batch size
    - 重复实验的次数——在args_test_speed.sh中修改
    - 是否跑nsys——在args_test_speed.sh中修改 默认不开启

- 确认完后，运行`bash resnet50_test.sh`


### 5. 处理结果数据，生成表格

- 运行时间数据：保存在data/路径下，命名为test_eager_commit_*_

  nsys文件：保存在data/路径下，命名为resnet50_eager_*.qdrep(默认不开启)


- 修改process_speed_data.py中的 /path/to/路径 和 nsys_root

  注: 如果跑nsys, 需要取消process_speed_data.py中对应的注释;

  运行`python3 process_speed_data.py --test_commits commit`

  将会在当前路径下生成一个process_res文件，存储markdown格式的表格

  这里目前可以运行resnet50_test.sh脚本一键生成;
