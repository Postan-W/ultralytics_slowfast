
def get_lines(filepath="./results.txt"):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.readlines()



def conf_filter(lines,conf_thre=0.7,classname="fall"):
    results = []
    for line in lines:
        conf = float(line.split(" ")[1].split(":")[1])
        if conf >= conf_thre and line.split(" ")[0] == classname:
            results.append(line)
    return results

lines = conf_filter(get_lines(),classname="fall")

#分析各个动作单独出现的情况
def single_action_prob(lines):
    action_dict = {}
    total_lines = len(lines)
    for line in lines:
        actions = line.strip().split(" ")[2:]
        for action in actions:
            name, prob = action.split(":")
            if name in action_dict:
                action_dict[name]["count"] += 1
                action_dict[name]["prob_sum"] += float(prob)
                if float(prob) < action_dict[name]["min_prob"]:
                    action_dict[name]["min_prob"] = float(prob)
                elif float(prob) > action_dict[name]["max_prob"]:
                    action_dict[name]["max_prob"] = float(prob)
            else:
                action_dict[name] = {"count":1, "prob_sum":float(prob),"min_prob":float(prob),"max_prob":float(prob)}

    for key in action_dict.keys():
        prob_avg = round(action_dict[key]["prob_sum"] / action_dict[key]["count"],3)
        action_dict[key]["prob_avg"] = prob_avg
        frequence = round(action_dict[key]["count"] / total_lines,3)
        action_dict[key]["frequence"] = frequence

    results = []
    for action in action_dict.keys():
        results.append({"name":action,**action_dict[action]})

    results = sorted(results,key=lambda x:x["frequence"],reverse=True)
    for result in  results:
        print("动作\"{}\"出现的频率是:{},平均预测概率是:{},最大预测概率是:{},最小预测概率是:{}".format(result["name"],result["frequence"],result["prob_avg"],result["max_prob"],result["min_prob"]))


single_action_prob(lines)

