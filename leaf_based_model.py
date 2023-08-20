import json
import os
import os.path
from datetime import datetime
import numpy as np
import pandas as pd

class LeafBasedModel:
    def __init__(self,
                 path_features_folder,
                 path_sentiment_folder,
                 path_company_names,
                 path_performance,
                 end_date,
                ):
        ###### PATHS PARAMETERS ############

        # path where to find the sentiment tables (earnings and complete)
        self.path_companies = path_company_names
        self.path_pickle = f"{path_features_folder}/df_stock_11bins_{end_date}_all.pkl"
        self.path_csv_sentiment = f"{path_sentiment_folder}{end_date}_freq_all.csv"
        self.path_performance_summary = f"{path_performance}performance_summary_values_returns.csv"
        self.path_performance_signal = f"{path_performance}signal_{end_date}.json"

        self.include_feature_parameters()
    def include_feature_parameters(self):
        #sentiment features
        self.feature_base_sentiment = [
            "sent_pos",
            "pos_sent_ratio",
            "sent_neg",
            "neg_sent_ratio",
            "all_count",
            "relevance",
        ]
        self.feature_base_sentiment_plot = [
            "Positive sentiment",
            "Positive share",
            "Negative sentiment",
            "Negative share",
            "News counts",
            "News relevance",
        ]
        #Technical features
        self.feature_base_tech = [
            "return past 7 days",
            "return past 14 days",
            "return past 30 days",
            "return past 60 days",
            "return past 90 days",
            "return past 180 days",
            "return past 365 days",
            "return past 14 days AVERAGE",
            "return past 30 days AVERAGE",
        ]
        # this list is only for plotting
        self.feature_base_tech_plot = [
            "Market return past 14 days",
            "Market return past 30 days",
        ]

        # sentiment past period (look back period of sentiment)
        self.periods = [30]

        #Features for model plotting
        self.features_model_plot = (
            self.feature_base_sentiment_plot + self.feature_base_tech_plot
        )

        # Decision tree parameters
        self.thresholds = [78, 83, 88]
        self.percentiles_return = [30, 70]

        # update period
        self.update_period = 300

        # Miscellaneous settings
        self.initial_shift = 1
        self.target_periods = [30]
        self.target_periods_samples = 20
        self.targets = [
            "return future " + str(i) + " days" for i in self.target_periods
        ]
        self.fee = 0.005
        self.fee_training = 0.01

        # companies to be removed
        self.comp_to_remove = ['Skan']

        # Periods filter and skip performance
        self.periods_filter = [[365, 30]]
        self.skip_performance = 15

        #Ensemble size
        self.ensemble_size_best = 15

    def compute_cap_weights(self):
        df_companies = pd.read_excel(self.path_companies, sheet_name="Sheet1")
        df_companies = df_companies.rename(columns={"Company identifier": "company"})

        df_companies = df_companies[~df_companies["company"].isin(self.comp_to_remove)]

        firms = df_companies["company"].tolist()
        weights_SPI = df_companies["weights_SPI"].tolist()

        # Define weights based on different criteria
        weights_SPI_normed = np.array(weights_SPI)

        w_all = weights_SPI_normed.copy()
        w_large = np.where(w_all < 4, 0, w_all)
        w_medium = np.where(np.logical_or(w_all > 10, w_all <= 0.2), 0, w_all)
        w_small_medium = np.where(w_all > 10, 0, w_all)
        w_small = np.where(w_all > 0.9, 0, w_all)

        # Define weight options
        weights_names = [
            "Equal weights",
            "All Cap",
            "Large Cap",
            "Medium Cap",
            "Small Cap",
            "Mid-Small Cap",
        ]

        weights_values = np.zeros((len(firms), len(weights_names)))
        weights_values[:, weights_names.index("Equal weights")] = np.ones(len(firms)) / len(firms)
        weights_values[:, weights_names.index("All Cap")] = w_all
        weights_values[:, weights_names.index("Large Cap")] = w_large
        weights_values[:, weights_names.index("Medium Cap")] = w_medium
        weights_values[:, weights_names.index("Small Cap")] = w_small
        weights_values[:, weights_names.index("Mid-Small Cap")] = w_small_medium

        # Store computed values
        self.weights_values = weights_values
        self.firms = firms
        self.firms_plot = firms.copy()
        self.weights_names = weights_names
        self.weights_SPI = weights_SPI
        print("Done with cap weights")

    def load_company_data(self, path_pickle, path_csv_sentiment):
        if not (os.path.exists(path_pickle) and os.path.exists(path_csv_sentiment)):
            return pd.DataFrame(None), True

        df = pd.read_pickle(path_pickle)
        df_sentiment = pd.read_csv(path_csv_sentiment).drop(columns="Unnamed: 0")
        df_sentiment.index = pd.to_datetime(df_sentiment["date"])

        df_merged = df.join(df_sentiment, how="inner")
        df = df_merged.copy()

        df = df[df["close"] > 0]
        if len(df) == 0:
            print("len(df) = 0 !")

        return df, False

    def run_performance(self, returns=None, buy_signal=None, fee=0.005, start_value=100, skip=1):
        ret = returns[::skip]
        buy = buy_signal[::skip]

        fees = np.where(np.abs(np.diff(np.append(1, buy))) > 0.1, fee, 0)
        buy_net = buy - fees

        prod = ret[:-1] * buy_net[:-1] + 1
        equity_cum = np.cumprod(prod)
        equity_cum = np.insert(equity_cum * start_value, 0, start_value)

        return equity_cum

    """
    def compute_threshold(self):
        print("Starting with compute_threshold")

        length_series = {}
        data_all = {}
        update_starts_global = {}

        for company in self.firms:
            print("Processing:", company)
            df, skip_bool = self.load_company_data(
                self.path_pickle.format(company),
                self.path_csv_sentiment.format(company),
            )

            if skip_bool:
                print("Skipping company process, company feature path does not exist.")
                continue

            df = df[self.initial_shift:]

            length = len(df[df.index.year < 2020]) + int(len(df[df.index.year == 2021]) * 2 / 12)
            length_series[company] = length

            if length == 0:
                print("Company has no data for processing:", company)
                continue

            update_starts = list(np.arange(0, length, self.update_period))
            update_starts.append(update_starts[-1] + self.update_period)
            update_starts_global[company] = [
                update + max(length_series.values()) - length for update in update_starts
            ]

            company_data = {}
            for per in self.periods:
                for i in self.feature_base_sentiment:
                    features = str(i) + str(per)
                    for threshold in self.thresholds:
                        buy_data = []
                        for update_start in update_starts_global[company]:
                            end = len(df) if update_start == update_starts_global[company][-1] else min(
                                update_start + self.update_period, len(df))
                            df20 = df.copy()[:end]
                            df_percentile = df20.copy()[:update_start]
                            buy = np.ones(len(df20))

                            if len(df_percentile[features].dropna()) > 1:
                                aid = np.percentile(
                                    df_percentile[features].dropna().values, threshold
                                )
                                buy[df20[features] > aid] = 0

                            buy_data.extend(buy[update_start:end])

                        col_name = f"{company} buy {features} {threshold}"
                        company_data[col_name] = pd.Series(buy_data, index=df20.index)

            for target_period, target in zip(self.target_periods, self.targets):
                company_data[f"{company} {target}"] = df[target]

            for feature in self.feature_base_tech:
                company_data[f"{company} {feature}"] = df[feature]

            company_data[f"{company} close"] = df["close"]

            for per in self.periods:
                for i in self.feature_base_sentiment:
                    features = str(i) + str(per)
                    company_data[f"{company} {features}"] = df[features]

            mods = [s for s in self.feature_base_tech if "AVERAGE" in s] + [
                s for s in self.feature_base_tech if "RSI" in s
            ]

            for per in self.periods:
                for i in mods:
                    features = str(i)
                    for threshold in self.thresholds:
                        buy_data = []
                        for update_start in update_starts_global[company]:
                            end = len(df) if update_start == update_starts_global[company][-1] else min(
                                update_start + self.update_period, len(df))
                            df20 = df.copy()[:end]
                            df_percentile = df20.copy()[:update_start]
                            buy = np.ones(len(df20))

                            if len(df_percentile[features].dropna()) > 1:
                                aid = np.percentile(
                                    df_percentile[features].dropna().values, 100 - threshold
                                )
                                buy[df20[features] < aid] = 0

                            buy_data.extend(buy[update_start:end])

                        col_name = f"{company} buy {features} {threshold}"
                        company_data[col_name] = pd.Series(buy_data, index=df20.index)

            data_all[company] = pd.DataFrame(company_data)

        self.data_all = data_all
        self.df = df
        self.update_starts_global = update_starts_global
        self.update_period_loc = int(max(length_series.values()) / (length_series[company] / self.update_period))
        self.length_series = length_series

        print("Done with compute_threshold")
        """

    def compute_historical_performance(self):
        # Ensure self.data_all is initialized before accessing it
        self.some_method_to_initialize_data_all()

        leaves = ["leaf1", "leaf2", "leaf3", "leaf4", "leaf5", "leaf6"]
        vars_kpi = ["excess return ratio all", "excess return ratio weighted"]

        company = list(self.data_all.keys())[0]
        all_columns_first_company = [
            s for s in self.data_all[company].columns if f"{company} buy" in s
        ]
        num_models = len(all_columns_first_company)
        update_starts = self.update_starts_global[list(self.data_all.keys())[0]]

        model_performance_all = np.zeros(
            (
                num_models,
                len(leaves),
                len(vars_kpi),
                len(self.firms),
                len(update_starts),
                len(self.periods_filter),
            )
        )

        for company in self.data_all:
            for ifilter in range(len(self.periods_filter)):
                self.data_all[company][
                    f"{company} {self.periods_filter[ifilter]} leaf index"
                ] = 0

        for company in self.data_all:
            self.data_all[company][f"{company} update index"] = 0
            update_starts = list(self.update_starts_global[company])
            for update_start in update_starts:
                ind0 = update_start
                ind1 = (
                    len(self.df) - 1
                    if update_start == update_starts[-1] or len(update_starts) <= 2
                    else min(update_start + update_starts[2] - update_starts[1], len(self.df) - 1)
                )
                self.data_all[company][f"{company} update index"][ind0:ind1] = update_starts.index(update_start)
                self.data_all[company][f"{company} update index"][-1] = np.max(
                    self.data_all[company][f"{company} update index"].values
                )

        for ifilter in range(len(self.periods_filter)):
            features_filter = [f"return past {i} days" for i in self.periods_filter[ifilter]]
            for company in self.data_all:
                update_starts = list(self.update_starts_global[company])
                length = self.length_series[company]
                start_data = max(0, int(update_starts[0]))
                cols_company = [s for s in self.data_all[company].columns if company in s]
                list_models = [s for s in self.data_all[company].columns if f"{company} buy" in s]
                df_col = self.data_all[company][cols_company]

                for update_start in update_starts:
                    end = len(df_col) if update_start == update_starts[-1] else int(
                        min(update_start + self.update_period_loc, len(df_col)))
                    df = df_col.copy()[start_data:end]
                    df_percentile = df.copy()[start_data:update_start]

                    if len(df_percentile[f"{company} {features_filter[1]}"].dropna()) > 1:
                        aid1 = np.percentile(df_percentile[f"{company} {features_filter[1]}"].dropna(),
                                             self.percentiles_return[0])
                        aid2 = np.percentile(df_percentile[f"{company} {features_filter[1]}"].dropna(),
                                             self.percentiles_return[1])
                    else:
                        aid1 = 0
                        aid2 = 0

                    ret365 = df[f"{company} {features_filter[0]}"]
                    ret30 = df[f"{company} {features_filter[1]}"]
                    ret_365_30 = (ret365 - ret30) / (1 + ret30)

                    for leaf in leaves:
                        mask = None
                        if leaf == "leaf1":
                            mask = (ret365 >= 0) & (ret30 >= aid2)
                        elif leaf == "leaf2":
                            mask = (ret365 >= 0) & (ret30 >= aid1) & (ret30 < aid2)
                        elif leaf == "leaf3":
                            mask = (ret365 >= 0) & (ret30 < aid1)
                        elif leaf == "leaf4":
                            mask = (
                                    (ret365 < 0)
                                    & (ret30 >= aid2)
                                    & ((df.index.year < 2020) | (df.index.year > 2020))
                            )
                        elif leaf == "leaf5":
                            mask = (
                                    (ret365 < 0)
                                    & (ret30 >= aid1)
                                    & (ret30 < aid2)
                                    & ((df.index.year < 2020) | (df.index.year > 2020))
                            )
                        elif leaf == "leaf6":
                            mask = (
                                    (ret365 < 0)
                                    & (ret30 < aid1)
                                    & ((df.index.year < 2020) | (df.index.year > 2020))
                            )

                        df.loc[
                            mask,
                            f"{company} {self.periods_filter[ifilter]} leaf index",
                        ] = leaves.index(leaf)
                        df_leaf = df[mask][:update_start]

                        if len(df_leaf) > 1:
                            mask2 = [False] * start_data + [mask[i] for i in range(len(mask))]
                            self.data_all[company].loc[
                                mask2,
                                f"{company} {self.periods_filter[ifilter]} leaf index",
                            ] = leaves.index(leaf)
                            w = [0.8, 1.2]

                            length = int(len(df_leaf) / 2)
                            ret1 = df_leaf[
                                       f"{company} {self.targets[self.target_periods.index(30)]}"
                                   ][:length]
                            ret2 = df_leaf[
                                       f"{company} {self.targets[self.target_periods.index(30)]}"
                                   ][length:]

                            num_ret1 = len(ret1[:: self.skip_performance])
                            num_ret2 = len(ret2[:: self.skip_performance])

                            hold_leaf1 = self.run_performance(
                                returns=ret1,
                                buy_signal=np.ones(len(ret1)),
                                skip=self.skip_performance,
                            )
                            hold_tot1 = hold_leaf1[-1] ** (12 / num_ret1)
                            hold_leaf2 = self.run_performance(
                                returns=ret2,
                                buy_signal=np.ones(len(ret2)),
                                skip=self.skip_performance,
                            )
                            hold_tot2 = hold_leaf2[-1] ** (12 / num_ret2)
                            hold_tot = hold_tot1 * hold_tot2

                            ratio_model = []
                            ratio_model_weighted = []
                            count_negative_hold = len(
                                df_leaf[
                                    df_leaf[
                                        f"{company} {self.targets[self.target_periods.index(30)]}"
                                    ]
                                    < 0
                                    ][:: self.skip_performance]
                            )

                            for i in list_models:
                                perf1 = self.run_performance(
                                    returns=ret1,
                                    buy_signal=df_leaf[i][:length],
                                    fee=self.fee_training,
                                    skip=self.skip_performance,
                                )
                                perf2 = self.run_performance(
                                    returns=ret2,
                                    buy_signal=df_leaf[i][length:],
                                    fee=self.fee_training,
                                    skip=self.skip_performance,
                                )
                                perf_tot1 = perf1[-1] ** (12 / num_ret1)
                                perf_tot2 = perf2[-1] ** (12 / num_ret2)

                                perf_tot = perf_tot1 * perf_tot2
                                ratio_model_weighted.append(
                                    (perf_tot1 / hold_tot1) ** w[0]
                                    * (perf_tot2 / hold_tot2) ** w[1]
                                )
                                ratio_model.append(perf_tot / hold_tot)

                            model_performance_all[
                            :,
                            leaves.index(leaf),
                            vars_kpi.index("excess return ratio all"),
                            self.firms.index(company),
                            update_starts.index(update_start),
                            ifilter,
                            ] = ratio_model

                            model_performance_all[
                            :,
                            leaves.index(leaf),
                            vars_kpi.index("excess return ratio weighted"),
                            self.firms.index(company),
                            update_starts.index(update_start),
                            ifilter,
                            ] = ratio_model_weighted

        self.num_models = num_models
        self.leaves = leaves

        self.ind_best_models = np.zeros(
            (
                self.ensemble_size_best,
                len(self.leaves),
                len(self.firms),
                len(
                    self.update_starts_global[
                        max(self.update_starts_global, key=lambda x: len(set(self.update_starts_global[x])))
                    ]
                ),
                len(self.periods_filter),
            )
        )

        self.model_performance_all = model_performance_all
        self.vars_kpi = vars_kpi

        print("Done with performance computation")

    def rank_models(self):
        kpi_vars = ["av performance", "av rank performance"]
        kpi_num = np.zeros((self.num_models, len(self.leaves), len(kpi_vars)))

        list_models = [
            s
            for s in self.data_all[self.firms[-1]].columns
            if str(self.firms[-1] + " buy") in s
        ]

        for ifilter in range(len(self.periods_filter)):
            for leaf in self.leaves:
                rank_perf = np.zeros(self.num_models)
                perf_av = np.zeros(self.num_models)

                for company in self.firms:
                    if company not in self.update_starts_global:
                        continue

                    update_starts = list(self.update_starts_global[company])

                    for update_start in update_starts:
                        perf = pd.DataFrame(
                            self.model_performance_all[
                            :,
                            self.leaves.index(leaf),
                            self.vars_kpi.index("excess return ratio all"),
                            self.firms.index(company),
                            update_starts.index(update_start),
                            ifilter,
                            ]
                        )
                        perf_av = perf_av + perf[0].values
                        ind = list(perf.sort_values(by=0, ascending=False).index)
                        rank = [ind.index(i) for i in range(len(list_models))]
                        rank_perf = rank_perf + rank

                        ind_to_use = ind[: self.ensemble_size_best] if len(ind) >= self.ensemble_size_best else ind[:]
                        self.ind_best_models[
                        : len(ind_to_use),
                        self.leaves.index(leaf),
                        self.firms.index(company),
                        update_starts.index(update_start),
                        ifilter,
                        ] = ind_to_use

                kpi_num[
                :, self.leaves.index(leaf), kpi_vars.index("av performance")
                ] = perf_av / len(self.firms)
                kpi_num[
                :, self.leaves.index(leaf), kpi_vars.index("av rank performance")
                ] = rank_perf / len(self.firms)

        print("Done with model ranking")

    def performance_test(self):
        print("starting with performance_test")
        data_all_performance_test = {}

        for company in self.data_all:
            data_all_performance_test[company] = pd.concat([
                self.data_all[company][::self.target_periods_samples],
                self.data_all[company][-1:]
            ])

        for company in self.firms:
            if company not in data_all_performance_test:
                continue

            close = data_all_performance_test[company][company + " close"]
            ret = [(close[i] / close[i - 1]) - 1 for i in range(1, len(close))]
            ret_net = np.append(ret, 0)
            data_all_performance_test[company][
                company + " " + "return future tailor days"
                ] = ret_net

        ensembles = [8]
        shares = [40]
        leaf_selects = ["all", "bear", "bull", "only leaf 5-6", "only leaf 6"]

        for ifilter in range(len(self.periods_filter)):
            for company in self.firms:
                print("performance_test", company)
                index_leaf = data_all_performance_test[company][
                    company + " " + str(self.periods_filter[ifilter]) + " leaf index"
                    ]
                list_models_loc = [
                    s
                    for s in data_all_performance_test[company].columns
                    if str(company + " buy") in s
                ]

                for ens in ensembles:
                    for share in shares:
                        buy_cum = []
                        for i in range(len(index_leaf)):
                            i_all = int(i * self.target_periods_samples)
                            if i_all < self.update_starts_global[company][0]:
                                buy = 0
                            else:
                                index_period = data_all_performance_test[company][
                                    company + " update index"
                                    ][i]
                                buy = 0
                                list_index = self.ind_best_models[
                                             :ens,
                                             index_leaf[i],
                                             self.firms.index(company),
                                             index_period,
                                             ifilter,
                                             ].astype(int)
                                for kk in list_index:
                                    buy = (
                                            buy
                                            + data_all_performance_test[company][
                                                list_models_loc[kk]
                                            ][i]
                                    )
                            buy_cum.append(buy)

                        buy_binary = np.array(buy_cum)
                        buy_binary[buy_binary <= ens * (100 - share) / 100] = 0
                        buy_binary[buy_binary > 1] = 1

                        for leaf_select in leaf_selects:
                            buy_binary_loc = buy_binary.copy()
                            if leaf_select == "bear":
                                buy_binary_loc[
                                    index_leaf <= 2
                                    ] = 1
                            elif leaf_select == "bull":
                                buy_binary_loc[
                                    index_leaf >= 3
                                    ] = 1
                            elif leaf_select == "only leaf 5-6":
                                buy_binary_loc[
                                    index_leaf <= 3
                                    ] = 1
                            elif leaf_select == "only leaf 6":
                                buy_binary_loc[
                                    index_leaf <= 4
                                    ] = 1

                            data_all_performance_test[company][
                                "buy "
                                + leaf_select
                                + " "
                                + str(self.periods_filter[ifilter])
                                + " "
                                + str(ens)
                                + " "
                                + str(share)
                                + " "
                                + company
                                ] = buy_binary_loc
                            data_all_performance_test[company][
                                "eq "
                                + leaf_select
                                + " "
                                + str(self.periods_filter[ifilter])
                                + " "
                                + str(ens)
                                + " "
                                + str(share)
                                + " "
                                + company
                                ] = self.run_performance(
                                returns=data_all_performance_test[company][
                                    company + " " + "return future tailor days"
                                    ].fillna(0),
                                fee=self.fee,
                                buy_signal=buy_binary_loc,
                            )

                        data_all_performance_test[company][
                            "buy hold " + company
                            ] = np.ones(len(data_all_performance_test[company]))
                        data_all_performance_test[company][
                            "eq hold " + company
                            ] = self.run_performance(
                            returns=data_all_performance_test[company][
                                company + " " + "return future tailor days"
                                ].fillna(0),
                            buy_signal=np.ones(len(buy_binary)),
                        )

        # compute porfolio performance

        models = []
        for ifilter in range(len(self.periods_filter)):
            for leaf_select in leaf_selects:
                for ens in ensembles:
                    for share in shares:
                        models.append(
                            leaf_select
                            + " "
                            + str(self.periods_filter[ifilter])
                            + " "
                            + str(ens)
                            + " "
                            + str(share)
                        )
        models.append("hold")
        print(models)

        for mod in models:
            for company in self.firms:
                variables = ["eq " + mod + " " + company]
                buys = ["buy " + mod + " " + company]

                data_all_performance_test[company][
                    "total " + mod
                    ] = data_all_performance_test[company][variables].mean(axis=1)
                data_all_performance_test[company]["total cash " + mod] = np.mean(
                    (
                            np.array(data_all_performance_test[company][variables])
                            * np.array((1 - data_all_performance_test[company][buys]))
                    ),
                    axis=1,
                )

                for weights_name in self.weights_names:
                    w = self.weights_values[:, self.weights_names.index(weights_name)]
                    tmp = np.array(data_all_performance_test[company][variables])
                    tmp[np.isnan(tmp)] = 0
                    mean = []
                    for i in range(np.shape(tmp)[0]):
                        data = tmp[i, :]
                        data[np.isnan(data)] = 0
                        w[np.isnan(data)] = 0
                        m = np.average(data, weights=w)
                        mean.append(m)
                    data_all_performance_test[company]["total " + weights_name + " " + mod] = mean

                    tmp = np.array(data_all_performance_test[company][variables]) * np.array(
                        (1 - data_all_performance_test[company][buys]))
                    tmp[np.isnan(tmp)] = 0
                    mean = []
                    for i in range(np.shape(tmp)[0]):
                        data = tmp[i, :]
                        data[np.isnan(data)] = 0
                        w[np.isnan(data)] = 0
                        m = np.average(data, weights=w)
                        mean.append(m)
                    data_all_performance_test[company]["total cash " + weights_name + " " + mod] = mean

        self.df_performance_test = data_all_performance_test
        self.column_to_extract = (
                "buy "
                + "bear"
                + " "
                + str(self.periods_filter[0])
                + " "
                + str(ensembles[0])
                + " "
                + str(shares[0])
                + " "
        )
        print("Done with performance test")

        return data_all_performance_test

    @staticmethod
    def create_monthly_signal(signal_dict):
        monthly_signal_dict = {}

        for company, date_signal_dict in signal_dict.items():
            monthly_signal_dict[company] = {}
            date_dict = {}
            for date in date_signal_dict:
                date_object = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
                year_month_string = date_object.strftime("%Y-%m")
                date_dict.setdefault(year_month_string, []).append(date)
            for year_month_string, date_list in date_dict.items():
                signal = 1.0
                for date in date_list:
                    if date_signal_dict[date] == 0:
                        signal = 0
                monthly_signal_dict[company][year_month_string] = signal
        return monthly_signal_dict

    def store(self):
        signal_dict = {}

        for company, df_company in self.df_performance_test.items():
            column_to_extract_store = self.column_to_extract + company
            tmp_feature = {
                str(feature_time): value
                for feature_time, value in df_company[column_to_extract_store].items()
            }
            signal_dict[company] = tmp_feature

        with open(self.path_performance_signal, "w") as json_file:
            json.dump(signal_dict, json_file, indent=2)

        monthly_signal_dict = self.create_monthly_signal(signal_dict)
        unique_list_times = sorted({date for values in monthly_signal_dict.values() for date in values})

        signal_dict_for_df = {"date": [str(date) for date in unique_list_times]}

        for company, df_company in self.df_performance_test.items():
            signal_dict_for_df[company] = []
            past_value = None
            for date in unique_list_times:
                if date in monthly_signal_dict[company]:
                    signal_dict_for_df[company].append(monthly_signal_dict[company][date])
                    past_value = monthly_signal_dict[company][date]
                elif past_value in (0, 1):
                    signal_dict_for_df[company].append(past_value)
                else:
                    signal_dict_for_df[company].append("")

        signal_df_to_store = pd.DataFrame.from_dict(signal_dict_for_df)
        signal_df_to_store.to_excel(self.path_performance_signal.replace(".json", ".xlsx"))

        var_table = [
            "weight", "return 20 years model", "return 20 years hold",
            "excess return 20 years", "yearly return model", "yearly return hold",
            "yearly excess return", "tracking error", "information ratio",
            "max drawdown model", "max drawdown hold"
        ]

        table_val = np.zeros((len(self.firms), len(var_table)))

        ens = 8
        share = 40
        mod = "bear [365, 30] "
        ratio = []
        return_model = []
        return_hold = []

        for company in self.firms:
            ind = self.firms.index(company)
            ret_20_model = (
                    self.df_performance_test[company][f"eq {mod}{ens} {share} {company}"].dropna()[-1] - 100
            )
            ret_20_hold = self.df_performance_test[company][f"eq hold {company}"][-1] - 100
            excess_ret_20 = ret_20_model - ret_20_hold
            ann_ret_model = (((ret_20_model + 100) / 100) ** (1 / 20) - 1) * 100
            ann_ret_hold = (((ret_20_hold + 100) / 100) ** (1 / 20) - 1) * 100
            ann_excess_ret = ann_ret_model - ann_ret_hold

            list_years = self.df_performance_test[company].index.year.unique()
            ret_model = [
                (((self.df_performance_test[company][f"eq {mod}{ens} {share} {company}"].dropna()[0] /
                   self.df_performance_test[company][f"eq {mod}{ens} {share} {company}"].dropna()[-1]) - 1) * 100)
                if len(self.df_performance_test[company][f"eq {mod}{ens} {share} {company}"].dropna()) > 0
                else None
                for year in list_years
            ]

            ret_hold = [
                (((self.df_performance_test[company][f"eq hold {company}"][0] /
                   self.df_performance_test[company][f"eq hold {company}"][-1]) - 1) * 100)
                if len(self.df_performance_test[company][f"eq hold {company}"]) > 0
                else None
                for year in list_years
            ]

            tracking_error = np.std(np.array(ret_model) - np.array(ret_hold))
            information_ratio = ann_excess_ret / tracking_error

            eq = self.df_performance_test[company][f"eq {mod}{ens} {share} {company}"].values
            eq_hold = self.df_performance_test[company][f"eq hold {company}"].values
            peak = peak_cum = dd = dd_cum = peak_hold = peak_cum_hold = dd_hold = dd_cum_hold = 0

            for i, eq_value in enumerate(eq):
                if eq_value >= peak:
                    peak = eq_value
                dd = (eq_value - peak) / peak
                dd_cum.append(dd)
                peak_cum.append(peak)

                eq_hold_value = eq_hold[i]
                if eq_hold_value >= peak_hold:
                    peak_hold = eq_hold_value
                dd_hold = (eq_hold_value - peak_hold) / peak_hold
                dd_cum_hold.append(dd_hold)
                peak_cum_hold.append(peak_hold)

            drawdown_hold = np.min(100 * np.array(dd_cum_hold))
            dd_cum = np.array(dd_cum)
            drawdown_model = np.min(100 * dd_cum[dd_cum > -100])

            table_val[ind, var_table.index("return 20 years model")] = np.int64(ret_20_model)
            table_val[ind, var_table.index("return 20 years hold")] = np.int64(ret_20_hold)
            table_val[ind, var_table.index("excess return 20 years")] = np.int64(excess_ret_20)
            table_val[ind, var_table.index("yearly return model")] = round(ann_ret_model, 1)
            table_val[ind, var_table.index("yearly return hold")] = round(ann_ret_hold, 1)
            table_val[ind, var_table.index("yearly excess return")] = round(ann_excess_ret, 1)
            table_val[ind, var_table.index("tracking error")] = round(tracking_error, 1)
            table_val[ind, var_table.index("information ratio")] = round(information_ratio, 2)
            table_val[ind, var_table.index("max drawdown model")] = int(drawdown_model)
            table_val[ind, var_table.index("max drawdown hold")] = int(drawdown_hold)

        w = self.weights_SPI / np.sum(self.weights_SPI) * 100
        table_val[:, var_table.index("weight")] = [round(weight, 3) for weight in w]
        df_table = pd.DataFrame(table_val, columns=var_table)
        df_table.index = self.firms

        df_table.to_csv(self.path_performance_summary)
        print("Saved file: " + self.path_performance_summary)


if __name__ == "__main__":
    base_folder = r"C:\Users\Farhan\Desktop\Project\unit_tests"
    model_config = {
        "path_performance": base_folder + r"\performance",
        "path_features_folder": base_folder + r"\price_data",
        "path_sentiment_folder": base_folder + r"\sent_write_path_tables_2022-12-31",
        "path_company_names": base_folder + r"\mapping_companies_unit_tests.xlsx",
    }

    LBM = LeafBasedModel(
        model_config["path_features_folder"],
        model_config["path_sentiment_folder"],
        model_config["path_company_names"],
        model_config["path_performance"],
        "end_date",
    )
    LBM.compute_cap_weights()
    #LBM.compute_threshold()
    LBM.compute_historical_performance()
    LBM.rank_models()
    df_companies_test = LBM.performance_test()
    LBM.store()