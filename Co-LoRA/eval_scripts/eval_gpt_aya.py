import argparse
import json
import os
import random

import openai
from openai import OpenAI
import time
import copy
import base64

NUM_SECONDS_TO_SLEEP = 0.01

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def get_eval(content: str, max_tokens: int):
    user_content = [
                        {"type":"text", "text":content['text']},
                    ]

    count = 0
    while count < 10:
        try:
            '''
            
            '''
            response = client.chat.completions.create(
                messages=[{
                    'role': 'system',
                    'content': "You are a helpful assistant, that rates models by the quality of their answers."
                },{
                    'role': 'user',
                    'content': user_content
                }],
                model="gpt-4o-mini-2024-07-18", # TODO: Choose gpt version
                temperature=0.2,  # TODO: figure out which temperature is best for evaluation
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except openai.APIConnectionError as e:
            print("The server could not be reached")
            print(e.__cause__)  # an underlying Exception, likely raised within httpx.
        except openai.RateLimitError as e:
            print("A 429 status code was received; we should back off a bit.")
        except openai.APIStatusError as e:
            print("Another non-200-range status code was received")
            print(e.status_code)
            print(e.response)
        time.sleep(NUM_SECONDS_TO_SLEEP)
        count += 1
    return "Score: 0"

def parse_score(review):
    try:
        score = review.split('[[')
        assert len(score) == 2
        score = score[-1].split(']]')[0]
        return int(score)
        # return score

    except Exception as e:
        print(e)
        print('error', review)
        return -1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    parser.add_argument('-r', '--result')
    parser.add_argument('-o', '--output')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--max-tokens', type=int, default=1024, help='maximum number of tokens produced in the output')
    args = parser.parse_args()
    print(f"start evaluating {args.result}")
    results_dict = json.load(open(args.result, 'r'))

    if os.path.isfile(os.path.expanduser(args.output)):
        cur_reviews = [json.loads(line) for line in open(os.path.expanduser(args.output))]
    else:
        output_dir = os.path.dirname(args.output)
        os.makedirs(output_dir, exist_ok=True)
        cur_reviews = []

    if os.path.exists(args.output):
        print('result exists')
        import sys
        sys.exit()
    
    review_file = open(f'{args.output}', 'a',encoding='utf-8')

    template = "[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. A good answer should follow these rules: \n1.It should be in the same language as the question. \n2. It should answer the request in the instruction. \n3.It should be factually and semantically comprehensible. \n4. It should be grammatically correct and fluent. \nBegin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".A human annotated answer is given for reference.\n\n[Question]\n{question}\n\n[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant's Answer]\n\n[Reference]\n{reference}"
    
    handles = []
    
    random.seed(args.random_seed)
    response_list = results_dict[:-1]
    # response_list = random.sample(response_list, 50)
    for item in response_list:
        review_item = copy.deepcopy(item)
        question = item['input'].split('<|eot_id|>')[0].split('<|start_header_id|>user<|end_header_id|>\n\n')[-1]
        prompt = template.format(question=question, answer=item['sentence'], reference=item['gt_sentence'])
        content = {
            "text": prompt,
        }

        review = get_eval(content, args.max_tokens)
        scores = parse_score(review)
        review_item['content'] = review
        review_item['score'] = scores
        cur_reviews.append(review_item)
        review_file.write(json.dumps(review_item, ensure_ascii=False) + '\n')
        review_file.flush()
    review_file.close()
    print('gpt evaluation finished!')