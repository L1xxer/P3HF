import argparse

from tqdm import tqdm
import pickle

import p3graph

log = p3graph.utils.get_logger()



def split(args):
    # 看一下toPKL.ipynb里的顺序！！
    sample_ids, speakers, labels, bigfive, gender, audios, videos, personality, train_vids, test_vids, dev_vids = pickle.load(open(args.src_dir, 'rb'))
    train, dev, test = [], [    ], []
    for vid in tqdm(train_vids, desc="train"):
        train.append(p3graph.Sample(vid, speakers[vid], labels[vid], bigfive[vid], audios[vid]["1"], audios[vid]["2"], audios[vid]["3"], videos[vid]["1"], videos[vid]["2"], videos[vid]["3"], personality[vid]))
        # 判断所有audio和video的长度是否一致
        len_a1 = len(audios[vid]["1"])
        len_a2 = len(audios[vid]["2"])
        len_a3 = len(audios[vid]["3"])
        len_v1 = len(videos[vid]["1"])
        len_v2 = len(videos[vid]["2"])
        len_v3 = len(videos[vid]["3"])
        if len_a1 != len_a2 or len_a1 != len_a3 or len_a1 != len_v1 or len_a1 != len_v2 or len_a1 != len_v3:
            log.error(f"Length mismatch for video {vid}: "
                      f"audio lengths: {len_a1}, {len_a2}, {len_a3}; "
                      f"video lengths: {len_v1}, {len_v2}, {len_v3}")
            continue
    for vid in tqdm(dev_vids, desc="dev"):
        dev.append(p3graph.Sample(vid, speakers[vid], labels[vid], bigfive[vid], audios[vid]["1"], audios[vid]["2"], audios[vid]["3"], videos[vid]["1"], videos[vid]["2"], videos[vid]["3"], personality[vid]))
    for vid in tqdm(test_vids, desc="test"):
        test.append(p3graph.Sample(vid, speakers[vid], labels[vid], bigfive[vid], audios[vid]["1"], audios[vid]["2"], audios[vid]["3"], videos[vid]["1"], videos[vid]["2"], videos[vid]["3"], personality[vid]))

        # print("test sample:", vid,  len(speakers[vid]), len(labels[vid]), len(audios[vid]["1"]), len(audios[vid]["2"]),
        #       len(audios[vid]["3"]), len(videos[vid]["1"]), len(videos[vid]["2"]),
        #       len(videos[vid]["3"]), len(personality[vid]))

    log.info("train vids:")
    log.info(sorted(train_vids))
    log.info("dev vids:")
    log.info(sorted(dev_vids))
    log.info("test vids:")
    log.info(sorted(test_vids))

    return train, dev, test


def main(args):
    train, dev, test = split(args)
    log.info("number of train samples: {}".format(len(train)))
    log.info("number of dev samples: {}".format(len(dev)))
    log.info("number of test samples: {}".format(len(test)))
    data = {"train": train, "dev": dev, "test": test}
    p3graph.utils.save_pkl(data, args.tag_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess.py")
    parser.add_argument("-s", "--src_dir", type=str, default="data/mpdd/MPDD_features.pkl")
    parser.add_argument("-t","--tag_dir", type=str, default="data/mpdd/ckpt/data.pkl")

    args = parser.parse_args()

    main(args)
