## Large Language Models Still Exhibit Bias in Long Text (ACL 2025)

This repository contains resources for **LTF-TEST** ([Paper](https://arxiv.org/abs/2410.17519)), a benchmark designed to evaluate bias in long-form text generation by large language models.

### Dataset

You can find the dataset in the `dataset/` folder:

- `LTF_TEST.jsonl`: Full prompt list  
- `Groups.json`: Metadata including demographic group labels

### Running the Evaluation

To evaluate models using LTF-TEST, simply run:

```bash
bash ex.sh
```

### 📌 Citation

If you find this work helpful, please consider citing:

```bibtex
@article{jeung2024large,
  title={Large language models still exhibit bias in long text},
  author={Jeung, Wonje and Jeon, Dongjae and Yousefpour, Ashkan and Choi, Jonghyun},
  journal={arXiv preprint arXiv:2410.17519},
  year={2024}
}
```

## Acknowledgements

This research was partially supported by AI Technology Development for Commonsense Extraction, Reasoning, and Inference from Heterogeneous Data (IITP) funded by the Ministry of Science and ICT (2022-0-00077).
