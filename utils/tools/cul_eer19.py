from evaluation import calculate_tDCF_EER



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--scoreFile', type=str, default="")
    args = parser.parse_args()
    # submit_file = 'log_eval/log_eval_7L_train_asvspoof2021eval_6_score.txt'

    def remove_duplicate_lines_inplace(file_path):
        # 用于存储已经遇到的第一个字符串
        seen_first_strings = set()

        # 打开文件进行读写
        with open(file_path, 'r+') as file:
            lines = file.readlines()  # 读取所有行

            # 将文件指针移到文件开头，准备写入新的内容
            file.seek(0)
            file.truncate()  # 清空文件内容

            for line in lines:
                # 提取每行的第一个字符串
                first_string = line.split()[0]

                # 如果第一个字符串没有重复，写入文件并添加到集合中
                if first_string not in seen_first_strings:
                    file.write(line)
                    seen_first_strings.add(first_string)

    # 调用函数，传入文件路径
    remove_duplicate_lines_inplace(args.scoreFile)
    output_file = args.scoreFile.replace("infer_19.log","a_eer19")
    
    cm_key_file="/data8/wangzhiyong/project/fakeAudioDetection/investigating_partial_pre-trained_model_for_fake_audio_detection/reference/fad/aasist/datasets/asvspoof2019/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt"
    calculate_tDCF_EER(args.scoreFile, cm_key_file,
                       output_file,printout=True)