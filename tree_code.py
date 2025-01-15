import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Находит оптимальный порог для разбиения вектора признака по критерию Джини.

    Критерий Джини определяется следующим образом:
    .. math::
        Q(R) = -\\frac {|R_l|}{|R|}H(R_l) -\\frac {|R_r|}{|R|}H(R_r),

    где:
    * :math:`R` — множество всех объектов,
    * :math:`R_l` и :math:`R_r` — объекты, попавшие в левое и правое поддерево соответственно.

    Функция энтропии :math:`H(R)`:
    .. math::
        H(R) = 1 - p_1^2 - p_0^2,

    где:
    * :math:`p_1` и :math:`p_0` — доля объектов класса 1 и 0 соответственно.

    Указания:
    - Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    - В качестве порогов, нужно брать среднее двух соседних (при сортировке) значений признака.
    - Поведение функции в случае константного признака может быть любым.
    - При одинаковых приростах Джини нужно выбирать минимальный сплит.
    - Для оптимизации рекомендуется использовать векторизацию вместо циклов.

    Parameters
    ----------
    feature_vector : np.ndarray
        Вектор вещественнозначных значений признака.
    target_vector : np.ndarray
        Вектор классов объектов (0 или 1), длина `feature_vector` равна длине `target_vector`.

    Returns
    -------
    thresholds : np.ndarray
        Отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно разделить на
        два различных поддерева.
    ginis : np.ndarray
        Вектор со значениями критерия Джини для каждого порога в `thresholds`.
    threshold_best : float
        Оптимальный порог для разбиения.
    gini_best : float
        Оптимальное значение критерия Джини.

    """
    # ╰( ͡☉ ͜ʖ ͡☉ )つ──☆*:・ﾟ   ฅ^•ﻌ•^ฅ   ʕ•ᴥ•ʔ

    # Сортируем вектор признаков и получаем индексы сортировки
    sorted_ind = np.argsort(feature_vector)
    # Сортируем сам вектор признаков по индексам
    feat_sort = feature_vector[sorted_ind]
    # Сортируем целевой вектор по тем же индексам
    target_sort = target_vector[sorted_ind]

    # Создаем маску для значений, которые отличаются от предыдущего (для определения порогов раздела)
    mask = feat_sort[1:] != feat_sort[:-1]

    # Рассчитываем возможные пороги для раздела (среднее значение между соседними элементами)
    threshold_vec = ((feat_sort[1:] + feat_sort[:-1]) / 2)[mask]

    # Общее количество элементов в целевом векторе
    AllSize = np.size(target_sort)

    # Создаем массив для размера левой части (количество элементов слева от порога)
    R_left_size = np.arange(1, AllSize)

    # Накопленная сумма по целевому вектору для левой части
    R_left_1 = np.cumsum(target_sort)

    # Вероятность класса 1 для левой части, считаем по накопленной сумме и размеру
    R_left_p1 = R_left_1[:-1] / R_left_size
    # Вероятность класса 0 для левой части
    R_left_p0 = 1 - R_left_p1

    # Накопленная сумма по правой части (правая часть = общий минус левая)
    R_right_1 = R_left_1[-1] - R_left_1[:-1]

    # Вероятность класса 1 для правой части
    R_right_p1 = R_right_1 / (AllSize - R_left_size)
    # Вероятность класса 0 для правой части
    R_right_p0 = 1 - R_right_p1

    # Вычисление индекса Джини для каждого возможного порога
    gini_vec = R_left_size / AllSize * (R_left_p0 ** 2 + R_left_p1 ** 2 - 1) + \
        (AllSize - R_left_size) / AllSize * (R_right_p0 ** 2 + R_right_p1 ** 2 - 1)

    # Фильтрация значений индекса Джини для корректных порогов (без одинаковых значений в feature_vector)
    gini_vec = gini_vec[mask]

    # Находим индекс порога с минимальным значением индекса Джини (наилучший порог)
    ind_best = np.argmax(gini_vec)
    # Находим сам лучший порог
    threshold_best = threshold_vec[ind_best]
    # Получаем индекс Джини для наилучшего порога
    gini_best = gini_vec[ind_best]

    # Возвращаем пороги, индексы Джини и лучший порог с его значением индекса Джини
    return threshold_vec, gini_vec, threshold_best, gini_best



class DecisionTree:
    def __init__(
        self,
        feature_types,
        max_depth=None,
        min_samples_split=None,
        min_samples_leaf=None,
    ):
        if any(ft not in {"real", "categorical"} for ft in feature_types):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        """
        Рекурсивное предсказание класса для одного объекта по узлу дерева решений.

        Если узел терминальный, возвращается предсказанный класс.
        Если узел не терминальный, выборка передается в соответствующее поддерево для дальнейшего предсказания.

        Parameters
        ----------
        x : np.ndarray
            Вектор признаков одного объекта.
        node : dict
            Узел дерева решений.

        Returns
        -------
        int
            Предсказанный класс объекта.
        """

        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():  # key - категория
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count  # по ключу лежит доля y=1 в категории
                sorted_categories = sorted(ratio.keys(),
                                           key=lambda k: ratio[k])  # отсортировали категории по доле единиц
                categories_map = dict(zip(sorted_categories,
                                          range(len(sorted_categories))))  # по названию категории получаем ее порядковый номер

                feature_vector = np.array([
                        categories_map[x] for x in sub_X[:, feature]])  # получили вектор порядковых номеров
            else:
                raise ValueError

            if np.all(feature_vector == feature_vector[0]):  # не получается разбить
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(  # список подходящих категорий
                            map(lambda x: x[0], # берем первый элемент, то есть ключ - исходную категорию
                                filter(
                                        lambda x: x[1] < threshold,
                                        categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self.depth += 1
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(
                sub_X[np.logical_not(split)],
                sub_y[np.logical_not(split)], node["right_child"])
    def _predict_node(self, x, node):
        """
        Рекурсивное предсказание класса для одного объекта по узлу дерева решений.

        Если узел терминальный, возвращается предсказанный класс.
        Если узел не терминальный, выборка передается в соответствующее поддерево для дальнейшего предсказания.

        Parameters
        ----------
        x : np.ndarray
            Вектор признаков одного объекта.
        node : dict
            Узел дерева решений.

        Returns
        -------
        int
            Предсказанный класс объекта.
        """
        # ╰( ͡☉ ͜ʖ ͡☉ )つ──☆*:・ﾟ   ฅ^•ﻌ•^ฅ   ʕ•ᴥ•ʔ
    
        if(node['type'] == 'terminal'):
            return node['class']
        else:
            feature_type = self._feature_types[node['feature_split']]
            if(feature_type == 'real'):
                if(x[node['feature_split']] < node['threshold']):
                    return self._predict_node(x, node['left_child'])
                else:
                    return self._predict_node(x, node['right_child'])
            else:
                if(x[node['feature_split']] in node['categories_split']):
                    return self._predict_node(x, node['left_child'])
                else:
                    return self._predict_node(x, node['right_child'])

    def fit(self, X, y):
        self.depth = 1
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)



