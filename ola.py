from HLL import HyperLogLog
from typing import List, Any
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np


class OLA:
    def __init__(self, widget: go.FigureWidget):
        """
            Base OLA class.

            *****************************************
            * You do not have to modify this class. *
            *****************************************

            @param widget: The dynamically updating plotly plot.
        """
        self.widget = widget

    def process_slice(df_slice: pd.DataFrame) -> None:
        """
            Process a dataframe slice. To be implemented in inherited classes.
        """
        pass

    def update_widget(self, groups_list: List[Any], values_list: List[Any]) -> None:
        """
            Update the plotly widget with newest groupings and values.

            @param groups_list: List of groups.
            @param values_list: List of grouped values (e.g., grouped means/sums).
        """
        self.widget.data[0]['x'] = groups_list
        self.widget.data[0]['y'] = values_list


class AvgOla(OLA):
    def __init__(self, widget: go.FigureWidget, mean_col: str):
        """
            Class for performing OLA by incrementally computing the estimated mean of *mean_col*.
            This class is implemented for you as an example.

            @param mean_col: column to compute filtered mean for.
        """
        super().__init__(widget)
        self.mean_col = mean_col

        # Bookkeeping variables
        self.sum = 0
        self.count = 0

    def process_slice(self, df_slice: pd.DataFrame) -> None:
        """
            Update the running mean with a data frame slice.
        """
        self.sum += df_slice.sum()[self.mean_col]
        self.count += df_slice.count()[self.mean_col]

        # Update the plot. The mean should be put into a singleton list due to Plotly semantics.
        # Note: there is no x axis label since there is only one bar.
        self.update_widget([""], [self.sum / self.count])


class FilterAvgOla(OLA):
    def __init__(self, widget: go.FigureWidget, filter_col: str, filter_value: Any, mean_col: str):
        """
            Class for performing OLA by incrementally computing the estimated filtered mean of *mean_col*
            where *filter_col* is equal to *filter_value*.

            @param filter_col: column to filter on.
            @param filter_value: value to filter for, i.e., df[df[filter_col] == filter_value].
            @param mean_col: column to compute filtered mean for.
        """
        super().__init__(widget)
        self.filter_col = filter_col
        self.filter_value = filter_value
        self.mean_col = mean_col

        # Put any other bookkeeping class variables you need here...
        self.filtered_sum = 0
        self.filtered_count = 0

    def process_slice(self, df_slice: pd.DataFrame) -> None:
        """
            Update the running filtered mean with a dataframe slice.
        """
        # Implement me!
        
        filtered_slice = df_slice[df_slice[self.filter_col] == self.filter_value]

        # Update the sum and count for filtered values
        self.filtered_sum += filtered_slice.sum()[self.mean_col]
        self.filtered_count += filtered_slice.count()[self.mean_col]

        est_mean = self.filtered_sum / self.filtered_count if self.filtered_count else 0

        self.update_widget([""], [est_mean])
        
        # Update the plot. The filtered mean should be put into a singleton list due to Plotly semantics.
        # hint: self.update_widget([""], *estimated filtered mean of mean_col*)


class GroupByAvgOla(OLA):
    def __init__(self, widget: go.FigureWidget, groupby_col: str, mean_col: str):
        """
            Class for performing OLA by incrementally computing the estimated grouped means of *mean_col*
            with *groupby_col* as groups.

            @param groupby_col: grouping column, i.e., df.groupby(groupby_col).
            @param mean_col: column to compute grouped means for.
        """
        super().__init__(widget)
        self.groupby_col = groupby_col
        self.mean_col = mean_col

        # Put any other bookkeeping class variables you need here...
        self.group_sums = {}
        self.group_counts = {}

    def process_slice(self, df_slice: pd.DataFrame) -> None:
        """
            Update the running grouped means with a dataframe slice.
        """
        # Implement me!

        # Group by the specified column and aggregate the sums and counts
        group_slice = df_slice.groupby(self.groupby_col)[self.mean_col].agg(['sum', 'count'])

        # Update the running sums and counts
        for group, data in group_slice.iterrows():
            # Initialize if group not in dictionary
            if group not in self.group_sums:
                self.group_sums[group] = 0
                self.group_counts[group] = 0

            # Update the sums and counts
            self.group_sums[group] += data['sum']
            self.group_counts[group] += data['count']

        # Calculate the new estimated means for each group
        est_mean = {
            group: self.group_sums[group] / self.group_counts[group]
            if self.group_counts[group] > 0 else 0
            for group in self.group_sums
        }

        list_of_groups = sorted(est_mean.keys())
        list_of_estimated_group_means = [est_mean[group] for group in list_of_groups]

        self.update_widget(list_of_groups, list_of_estimated_group_means)

        # hint: self.update_widget(*list of groups*, *list of estimated group means of mean_col*)


class GroupBySumOla(OLA):
    def __init__(self, widget: go.FigureWidget, original_df_num_rows: int, groupby_col: str, sum_col: str):
        """
        Class for performing OLA by incrementally computing the estimated grouped sums of *sum_col*
        with *groupby_col* as groups.

        @param original_df_num_rows: number of rows in the original dataframe before sampling and slicing.
        @param groupby_col: grouping column, i.e., df.groupby(groupby_col).
        @param sum_col: column to compute grouped sums for.
        """
        super().__init__(widget)
        self.original_df_num_rows = original_df_num_rows
        self.groupby_col = groupby_col
        self.sum_col = sum_col
        self.group_sums = {}
        self.total_processed_rows = 0 

    def process_slice(self, df_slice: pd.DataFrame) -> None:
        """
            Update the running filtered mean with a dataframe slice.
        """
        self.total_processed_rows += len(df_slice)
        
        group_slice = df_slice.groupby(self.groupby_col)[self.sum_col].sum()

        for group, group_sum in group_slice.items():
            self.group_sums[group] = self.group_sums.get(group, 0) + group_sum

        scale_factor = self.original_df_num_rows / self.total_processed_rows

        scaled_sums = {group: sum_value * scale_factor for group, sum_value in self.group_sums.items()}

        sorted_groups = sorted(scaled_sums.keys())
        sorted_sums = [scaled_sums[group] for group in sorted_groups]

        self.update_widget(sorted_groups, sorted_sums)
        
        # Update the plot
        # hint: self.update_widget(*list of groups*, *list of estimated grouped sums of sum_col*)

class GroupByCountOla(OLA):
    def __init__(self, widget: go.FigureWidget, original_df_num_rows: int, groupby_col: str, count_col: str):
        super().__init__(widget)
        self.original_df_num_rows = original_df_num_rows
        self.groupby_col = groupby_col
        self.count_col = count_col
        self.group_counts = {}
        self.total_processed_rows = 0

    def process_slice(self, df_slice: pd.DataFrame) -> None:
        """
            Update the running filtered mean with a dataframe slice.
        """
        self.total_processed_rows += len(df_slice)
        group_counts_slice = df_slice.groupby(self.groupby_col)[self.count_col].count()

        for group, count in group_counts_slice.items():
            if group not in self.group_counts:
                self.group_counts[group] = 0
            self.group_counts[group] += count

        scale_factor = self.original_df_num_rows / self.total_processed_rows
        scaled_counts = {group: count * scale_factor for group, count in self.group_counts.items()}

        sorted_groups = sorted(scaled_counts.keys())
        sorted_counts = [scaled_counts[group] for group in sorted_groups]

        self.update_widget(sorted_groups, sorted_counts)



class FilterDistinctOla(OLA):
    def __init__(self, widget: go.FigureWidget, filter_col: str, filter_value: Any, distinct_col: str):
        """
        Class for performing OLA by incrementally computing the estimated cardinality (distinct elements) *distinct_col*
        where *filter_col* is equal to *filter_value*.

        @param filter_col: column to filter on.
        @param filter_value: value to filter for, i.e., df[df[filter_col] == filter_value].
        @param distinct_col: column to compute cardinality for.
        """
        super().__init__(widget)
        self.filter_col = filter_col
        self.filter_value = filter_value
        self.distinct_col = distinct_col

        self.hll = HyperLogLog(p=2, seed=123456789)

    def process_slice(self, df_slice: pd.DataFrame) -> None:
        """
        Update the running filtered cardinality with a dataframe slice.
        """
        # Filter the slice based on filter_col and filter_value
        filter_slice = df_slice[df_slice[self.filter_col] == self.filter_value]
        
        # Iterate over the distinct_col in the filtered slice, convert each value to string, and add to HLL
        for value in filter_slice[self.distinct_col].unique():
            self.hll.add(str(value))

        est_cardinality = self.hll.cardinality()

        self.update_widget([""], [est_cardinality])

        # The self.update_widget call above assumes that the widget is designed to display a single value.
        # The [""] is a placeholder for the x-axis (which doesn't apply here since we're just showing a single value),
        # and the [estimated_cardinality] is the y-axis value, showing the estimated number of distinct elements.

        # Update the plot. The filtered cardinality should be put into a singleton list due to Plotly semantics.
        # hint: self.update_widget([""], *estimated filtered cardinality of distinct_col*)