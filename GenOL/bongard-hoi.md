Bongard-HOI

HoI는 그때 급하게 하느라 좀 주먹구구식으로 한 부분이 있어서 documentation으로 하기가 어려운 측면이 있음.

1. 처음에는 train.json (MA) 개수를 그대로 생성하는 걸 목표로 했음
따라서 아래 directory 구조로 image를 생성하고 generate_train.py를 써서 json 파일을 생성함.
(web으로 이미지 생성한걸로 json 만들려면 이렇게 하면 될듯?)

hoi_diversified_sdxl/
    airplane/
        exit/
            image_000000.png
            image_000001.png
            ...
        inspect/
            image_000000.png
            image_000001.png
            ...
        ...
    bicycle/
    ...
그리고 나면 train.json이랑 완전히 똑같은 형태의 json이 만들어지므로 문제없이 돌릴 수 있음


2. 그런데 이렇게 했을 때 generated image에서는 미리 negative를 다 알고 있다는 가정이 말도 안된다?
이런 문제가 생겨서 좀 다른 방법으로 해보기로 함.

generate_train_gpt.py 사용
train.json에서 id 하나하나 가져와서 positive action, negative action list 가져오고,
gpt api에 prompt 넣어서 7개의 hard-negative action 뽑고,
뽑힌 positive action, negative action 써서 새로운 json 파일 생성
(기존에 이미지 생성한걸 재활용해야 돼서 이런 방법으로 했던 걸로 기억)


3. 그리고 나서 이유는 잘 기억 안나는데 최종적으로 또 방식을 바꿈.
아마 train.json에 해당하는 이미지를 처음부터 전부 생성하고 RMD 하면 너무 많아서, ./ma_splits/Bongard-HOI_train_seed1.json 이런
파일을 기준으로 생성하기로 했던 것 같은데?

prompt_generation/generate_prompts_hoi_diversified_new.py 사용
일단 ./ma_splits/Bongard-HOI_train_seed1.json 불러와서 iteration 돌면서,
positive action에 대한 prompt 생성하고,
negative action들 list에서 negative action 7개인가 뽑고,
뽑은 negative action 넣어서 또 prompt 생성하고,
그 다음에 hoi_diversified_new.json 이런 식으로 저장

그리고 나면 ./ma_splits/Bongard-HOI_train_seed1.json과 동일하게 이미지를 생성할 수 있는 prompt가 만들어진 것.
이걸로 이미지 생성하면 구조는 아래와 같음 (50456개)
0/
    pos/
        image_000000.png
        image_000001.png
        image_000002.png
        image_000003.png
        image_000004.png
        image_000005.png
        image_000006.png
    neg/
        image_000000.png
        image_000001.png
        image_000002.png
        image_000003.png
        image_000004.png
        image_000005.png
        image_000006.png
1/
    pos/
        image_000000.png
        ...
    neg/
        image_000000.png
        ...

이걸로 hoi image들 전부 생성한 다음에 RMD score 계산하고 ensemble 한걸로 기억