import os
import zipfile

# PyTorch 1.1 moves _download_url_to_file
#   from torch.utils.model_zoo to torch.hub
# PyTorch 1.0 exists another _download_url_to_file
#   2 argument
# TODO: If you remove support PyTorch 1.0 or older,
#       You should remove torch.utils.model_zoo
#       Ref. PyTorch #18758
#         https://github.com/pytorch/pytorch/pull/18758/commits
try:
    from torch.utils.model_zoo import _download_url_to_file
except ImportError:
    try:
        from torch.hub import download_url_to_file as _download_url_to_file
    except ImportError:
        from torch.hub import _download_url_to_file


def unzip(source_filename, dest_dir):
    with zipfile.ZipFile(source_filename) as zf:
        zf.extractall(path=dest_dir)


if __name__ == '__main__':
    _download_url_to_file('https://www.dropbox.com/s/lrvwfehqdcxoza8/saved_models.zip?dl=1', 'saved_models.zip', None, True)
    unzip('saved_models.zip', '.')

'''
在这段Python代码中，首先尝试从不同的位置导入`_download_url_to_file`函数，这是用于从网络地址下载文件的PyTorch相关函数。根据不同的PyTorch版本，`_download_url_to_file`函数可能位于不同的模块中。随后的代码使用这个函数下载了一个名为`saved_models.zip`的文件，并将其解压到了当前执行脚本的目录中（由 `'.'` 指定）。

具体来说：

1. **下载文件的位置**：
   - 调用`_download_url_to_file`函数时，指定的目标文件名为`'saved_models.zip'`。这意味着下载的文件将保存在脚本运行的当前目录中，文件名为`saved_models.zip`。
   
2. **解压的位置**：
   - 调用`unzip`函数将这个ZIP文件解压到了当前目录（`.`）。这表示所有解压出来的文件和文件夹将直接位于脚本所在的目录。

总结一下，如果你运行这个脚本，你将在脚本所在的目录中找到一个名为`saved_models.zip`的压缩文件，以及这个压缩文件解压后的所有内容。如果你想查看这些文件，可以在脚本所在的目录中使用文件浏览器或在终端中使用命令（如`ls`或`dir`，取决于你的操作系统）查看。
'''