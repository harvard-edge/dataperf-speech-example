import os

import fire

import constants as c
import utils as utils
import eval as eval


def main(setup_yaml_path=c.DEFAULT_SETUP_YAML_PATH):
    task_setup = utils.load_yaml(setup_yaml_path)
    data_dir = task_setup[c.SETUP_YAML_DATA_DIR_KEY]
    dim = task_setup[c.SETUP_YAML_DIM_KEY]
    emb_path = os.path.join(data_dir, task_setup[c.SETUP_YAML_EMB_KEY])

    ss = utils.get_spark_session(task_setup[c.SETUP_YAML_SPARK_MEM_KEY])

    print('\nLoading embeddings...', end='')
    emb_df = utils.load_emb_df(ss=ss, path=emb_path, dim=dim)
    print('Done\n')

    task_paths = {
        task: task_setup[task] for task in task_setup[c.SETUP_YAML_TASKS_KEY]}
    task_scores = {}
    for task, paths in task_paths.items():
        print(f'Evaluating task: {task}')
        train_path, test_path = [os.path.join(data_dir, p) for p in paths]

        print(f'Loading training data for {task}...', end='')
        train_df = utils.load_train_df(ss=ss, path=train_path)
        train_df = utils.add_emb_col(df=train_df, emb_df=emb_df)
        print('Done')

        print(f'Loading test data for {task}...', end='')
        test_df = utils.load_test_df(ss=ss, path=test_path, dim=dim)
        print('Done')

        print(f'Training classifier for {task}...', end='')
        clf = eval.get_trained_classifier(df=train_df)
        print('Done')

        print(f'Scoring trained classifier for {task}...', end='')
        task_scores[task] = eval.score_classifier(df=test_df, clf=clf)
        print('Done\n')

    save_dir = os.path.join(data_dir, task_setup[c.SETUP_YAML_RESULTS_KEY])
    utils.save_results(data=task_scores, save_dir=save_dir, verbose=True)


if __name__ == "__main__":
    fire.Fire(main)
