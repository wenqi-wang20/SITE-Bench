import json


def calculate_parts_performance(img_results, video_results):
    
    # load jsonl files combined (from lmms-eval log file, not json file)
    all_results = []
    with open(img_results, 'r', encoding='utf-8') as f:
        for line in f:
            all_results.append(json.loads(line))
    with open(video_results, 'r', encoding='utf-8') as f:
        for line in f:
            all_results.append(json.loads(line))
            
    # calculate the chance adjusted accuracy for each category and the overall
    categories = {}
    overall = {"acc": 0, "total": 0}
    for result in all_results:
        # get the category of the item
        category = result["doc"]["category"]
        # get the accuracy of the item
        accuracy = result["accuracy"]["overall"]
        # get the number of options
        num_options = len(result["doc"]["options"])
        # calculate the chance adjusted accuracy
        chance_adjusted_accuracy = accuracy - 1.0 / num_options
        
        # for category performance
        if category not in categories:
            categories[category] = {"acc": 0, "total": 0}
        categories[category]["acc"] += chance_adjusted_accuracy
        categories[category]["total"] += 1 - 1.0 / num_options
        
        # for overall performance
        overall["acc"] += chance_adjusted_accuracy
        overall["total"] += 1 - 1.0 / num_options
        
    
    performance = {}
    performance["overall"] = overall["acc"] / overall["total"] * 100
    print(f"Overall: {performance['overall']:.2f}")
    
    for category in categories:
        performance[category] = categories[category]["acc"] / categories[category]["total"] * 100
        # print the performance of each category
        print(f"{category}: {performance[category]:.2f}")
        

    return performance


if __name__ == "__main__":
    img_results = "img_results.jsonl"       # replace with the actual path to the site_bench_image results jsonl file
    video_results = "video_results.jsonl"   # replace with the actual path to the site_bench_video results jsonl file
    calculate_parts_performance(img_results, video_results)