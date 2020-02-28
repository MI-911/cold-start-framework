

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

    def interview(self, users):
        """
        Predicts a score for all items given a user.
        :param users: A dictionary of the following format:
            {
                userId : {
                    validation: {
                        negative: [ item_ids ]
                        positive: item_id
                    },
                    sets: [
                        answers: { item_id: sentiment },
                        negative: [ item_ids ],
                        positive: item_id
                    ]
                }
            }
        :return: A dictionary of the following format:
            {
                item_id : score  // Best fitting item --> highest score
            }
        """
        raise NotImplementedError

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

