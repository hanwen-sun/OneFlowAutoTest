import os
import glob
import yaml
from extract_config import get_args

# extract result from libai log file
def extract_info_from_file(log_file):
    result_dict = {}
    result_dict["samples"] = 0
    result_dict["memory"] = 0
    #print("come here" + log_file)
    with open(log_file, "r") as f:
        for line in f.readlines():
            if "iteration:" in line and "time:" in line:
                ss = line.split(" ")
                iteration_index = ss.index("iteration:")
                iteration_number = int(ss[iteration_index + 1].strip().split("/")[0])
                time_index = ss.index("time:")                
                samples = float(ss[time_index + 8])
                #samples = float(ss[time_index + 1].strip().split("(")[1][:-1])
                if iteration_number == 219:
                    result_dict["samples"] = samples
            elif "MiB," in line and "utilization" not in line:
                ss = line.split(" ")
                #print(ss)
                if ss[-1] == 'MiB\n':
                    memory_userd = int(ss[-2])
                    #print(memory_userd)
                    if (
                        "memory" not in result_dict.keys()
                        or result_dict["memory"] < memory_userd
                    ):
                        result_dict["memory"] = memory_userd
    return result_dict


def get_config(yaml_file):
    with open("{}/config.yaml".format(yaml_file), "r") as f:
        config_data = yaml.full_load(f)

    return config_data

# extract result from megatron log file
def megatron_extract(log_file):
    result_dict = {}
    result_dict["samples"] = 0
    result_dict["memory"] = 0
    with open(log_file, "r") as f:
        for line in f.readlines():
            if "consumed samples:" in line:
                # print('iteration: ' + line)
                ss = line.split(" ")
                #print(ss)
                ss = [item for item in ss if item.strip()]
                #print(ss)
                iteration_index = ss.index("iteration")
                iteration_number = int(ss[iteration_index + 1].strip().split("/")[0]) # ?
                #print(iteration_number)
                consumed_sample_index = ss.index("samples:")
                consumed_samples = int(ss[consumed_sample_index + 1])
                #print(consumed_samples)
                time_index = ss.index("(ms):")
                time = float(ss[time_index + 1])
                #print(time)
                if iteration_number == 200:
                    result_dict["samples"] = "{:.2f}".format((consumed_samples / 200) * (1000 / time))
            elif "MiB," in line and "utilization" not in line:
                #print('memory: ' + line)
                ss = line.split(" ")
                memory_userd = int(ss[-2])
                if (
                    "memory" not in result_dict.keys()
                    or result_dict["memory"] < memory_userd
                ):
                    result_dict["memory"] = memory_userd
    return result_dict

# write the result
def extract_result(args, extract_func):
    megatron_list = glob.glob(os.path.join(args.compare_log, "*/*.log"))
    megatron_list = sorted(megatron_list)

    megatron_throughput_final_result_dict = {}
    for m_l in megatron_list:
        megatron_result_dict = megatron_extract(m_l)
        #print(megatron_result_dict)
        tmp_file_name = m_l.split("/")
        #print(tmp_file_name)
        case_header = "_".join(tmp_file_name[-1].split("_")[1:-2]).lower()
        #print(megatron_throughput_final_result_dict.keys())
        if case_header not in megatron_throughput_final_result_dict.keys():
            megatron_throughput_final_result_dict[case_header] = {}
        megatron_throughput_final_result_dict[case_header]["case_name"] = case_header
        megatron_throughput_final_result_dict[case_header][
            "megatron_memory"
        ] = megatron_result_dict["memory"]

        megatron_throughput_final_result_dict[case_header][
            "megatron_samples"
        ] = megatron_result_dict["samples"]
        #megatron_throughput_final_result_dict[case_header][
        #    "megatron_log"
        #] = "https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base_supple/full/{}".format(
        #    "/".join(tmp_file_name[-2:])
        #)
        #print(megatron_throughput_final_result_dict)

    logs_list = glob.glob(os.path.join(args.test_log, "*/*/output.log"))
    logs_list = sorted(logs_list)
    throughput_final_result_dict = {}
    markdown_table_header = """

|      | [libai-%s](https://github.com/Oneflow-Inc/oneflow/tree/%s) | [Megatron](https://github.com/NVIDIA/Megatron-LM/commit/e156d2fea7fc5c98e645f7742eb86b643956d840)                                                     |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |""" % (args.oneflow_commit, args.oneflow_commit)
    markdown_table_body = """
| {case_name}   | [{libai_memory}] MiB/[{libai_samples}] samples/s                                 | {megatron_memory} MiB/[{megatron_samples}] samples/s |"""
    # print(logs_list[0])
    tmp_markdown_table_header = markdown_table_header.format(
        logs_list[0].split("/")[-4], logs_list[0].split("/")[-4],
    )

    for l in logs_list:
        libai_result_dict = extract_func(l)
        tmp_file_name = l.split("/")
        # case_config = get_config("/".join(tmp_file_name[:-1]))
        case_header = "_".join(tmp_file_name[-2].split("_")[1:-2]).lower()

        if case_header not in throughput_final_result_dict.keys():
            throughput_final_result_dict[case_header] = {}
        throughput_final_result_dict[case_header]["case_name"] = case_header
        throughput_final_result_dict[case_header]["libai_memory"] = libai_result_dict[
            "memory"
        ]
        throughput_final_result_dict[case_header]["libai_samples"] = libai_result_dict[
            "samples"
        ]
        #throughput_final_result_dict[case_header][
        #    "libai_log"
        #] = "https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/{}/{}".format(args.oneflow_commit,
        #    "/".join(tmp_file_name[-3:])
        #)
        throughput_final_result_dict[case_header]["megatron_memory"] = 0
        throughput_final_result_dict[case_header]["megatron_samples"] = 0
        throughput_final_result_dict[case_header]["megatron_log"] = ""
        #print('case_header:' + case_header)
        #print(megatron_throughput_final_result_dict.keys())
        if case_header in megatron_throughput_final_result_dict.keys():
            throughput_final_result_dict[case_header][
                "megatron_memory"
            ] = megatron_throughput_final_result_dict[case_header]["megatron_memory"]
            throughput_final_result_dict[case_header][
                "megatron_samples"
            ] = megatron_throughput_final_result_dict[case_header]["megatron_samples"]
            #throughput_final_result_dict[case_header][
            #    "megatron_log"
            #] = megatron_throughput_final_result_dict[case_header]["megatron_log"]
            #print(throughput_final_result_dict[case_header]["megatron_memory"])
            #print(throughput_final_result_dict[case_header]["megatron_samples"])
            #print(markdown_table_body.format(**throughput_final_result_dict[case_header]))
        tmp_markdown_table_header += markdown_table_body.format(
            **throughput_final_result_dict[case_header]
        )

    with open("{}/dlperf_result.md".format(args.test_log), "w",) as f:
        f.write(tmp_markdown_table_header)


if __name__ == "__main__":
    args = get_args()
    extract_result(args, extract_info_from_file)
