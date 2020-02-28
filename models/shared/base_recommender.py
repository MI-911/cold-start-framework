

class RecommenderBase:

    def warmup(self, training):
        """
        Fits the model to the training data.
        :param training: A dictionary of the following format:
            {
                userId : {
                    training: {
                        entityId: sentiment
                    },
                    validation: {
                        negative: [ item_ids ],
                        positive: item_id
                    }
                }
            }
        :return: None
        """
        raise NotImplementedError

    def interview(self, answers, max_n_questions=5):
        """
        Predicts the next question(s) to ask a user
        :param answers: A state dictionary of the following format:
            {
                item_id: sentiment
            }
        :param max_n_questions: The maximally allowed length of the returned list.
        :return: A list of the following format:
            [
                item_ids  // Next question(s) to ask the user
            ]
        """
        raise NotImplementedError

    def predict(self, items, answers):
        """
        Predicts the ranking of the provided items given a user's answers.
        :param items: A list of the following format:
            [
                item_ids  // The items to rank
            ]
        :param answers: A state dictionary of the following format:
            {
                item_id: sentiment
            }
        :return: A dictionary of the following format:
            {
                item_id: score  // The best fitting item gets the highest score.
            }
        """

    def get_params(self):
        """
        Returns the model's current parameters so they can be reused later.
        :return: A dictionary of { param_key: value } key-value pairs.
        """
        raise NotImplementedError

    def load_params(self, params):
        """
        Loads the provided model parameters. Parameters should be provided from an outside source (e.g. the runner).
        :param params: A dictionary of { param_key: value } key-value pairs.
        """
        raise NotImplementedError
