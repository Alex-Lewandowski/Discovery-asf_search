from datetime import datetime
import logging

import asf_search as asf
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import shape
from tqdm import tqdm

tqdm.pandas()

"""
    TODOs:
      - stack length getter
      - support add/remove pairs
      - add optional overlap threshold
      - optionally connect both beginning and end of each season with 1-year pairs (currently connects ends only)
      - optional arg for target date from which to create bridge pairs
      - optional automatic disconnected-stack correction 
        - If there is an interruption in data at the end of a season, a disconnection can occur
          - work backwards from the end of the season until a scene is found that can connect to the following year
        - if the final season in a stack contains no scene within the temporal baseline threshold of the end of the season, it 
          will be disconnected
          - solution: perform backwards search to connect end of stack to previous year (maintain earlier date as ref scene)
"""


class ASFSBASStack:
    """
    The ASFSBASStack class creates connected SBAS stacks from Sentinel-1 SLCs and Sentinel-1 SLC bursts. All pairs are created based
    on the currently set temporal and perpendicular baselines. If a season is defined, out-of-season acquisitions are filtered and
    seasonal gaps are bridged with 1-year pairs. To determine the number of seasonal bridge pairs to create, a quantity is estimated
    by dividing the temporal baseline by an expected repeat pass frequency for the satellite (12 days for Sentinel-1). If a
    reference scene is expected to have 3 pairs, and one is available in-season, only 2 out-of-season pairs will be created. 

    Stacks are not guaranteed to be connected. A lack of data acquired at a given time and place can cause gaps.

    Key methods:
        - `plot()`
        - `get_insar_pairs()`

    """
                
    def __init__(self, **kwargs: dict):
        self.plot_available = False
        try:
            import plotly.graph_objects as go
            import networkx as nx
            self.plot_available = True
        except ImportError:
            print('Warning: ASFSBASStack.plot() requires additional dependencies: plotly and/or networkx')
        
        self._ref_scene_id = kwargs.get('refSceneName', None)  
        self._season = kwargs.get('season', ('1-1', '12-31'))
        self._start = kwargs.get('start', None)
        self._end = kwargs.get('end', None)
        self._needs_ref_stack_update = True
        self._sbas_stack = None

        self.perp_baseline = kwargs.get('perpendicularBaseline', 400)
        self.temporal_baseline = kwargs.get('temporalBaseline', 36)
        self.repeat_pass_freq = kwargs.get('repeatPassFrequency', 12)

        # build stack upon initialization if min required args were passed
        if self._ref_scene_id and self._end:
            self._sbas_stack = self.build_sbas_stack()
            self._needs_ref_stack_update = False

    
    @property
    def ref_scene_id(self):
        return self._ref_scene_id

    
    @ref_scene_id.setter
    def ref_scene_id(self, scene_id: str):
        self._ref_scene_id = scene_id
        self._needs_ref_stack_update = True

    
    @property
    def season(self):
        return self._season

    
    @season.setter
    def season(self, season: tuple[str,str]):
        self._season = season
        self._needs_ref_stack_update = True

    
    @property
    def start(self):
        return self._start

    
    @start.setter
    def start(self, stack_start_date: str):
        if (
            hasattr(self, '_sbas_stack') and 
            not (
                pd.to_datetime(self._sbas_stack['stopTime']).min() <=
                pd.to_datetime(stack_start_date) <=
                pd.to_datetime(self._sbas_stack['stopTime']).max()
            )
        ):
            self._needs_ref_stack_update = True
        self._start = stack_start_date

    
    @property
    def end(self):
        return self._end

    
    @end.setter
    def end(self, stack_end_date: str):
        if (
            hasattr(self, '_sbas_stack') and 
            not (
                pd.to_datetime(self._sbas_stack['stopTime']).min() <=
                pd.to_datetime(stack_end_date, utc=True) <=
                pd.to_datetime(self._sbas_stack['stopTime']).max()
            )
        ):
            self._needs_ref_stack_update = True
        self._end = stack_end_date

    
    @property
    def sbas_stack(self):
        if self._needs_ref_stack_update:
            self._sbas_stack = self.build_sbas_stack()
            self._needs_ref_stack_update = False
        else:
            self._sbas_stack['insarNeighbors'] = self._sbas_stack.apply(
            lambda row: self.get_seasonal_nearest_neighbors(self._sbas_stack, row['sceneName']),
            axis=1
        )
        return self._sbas_stack

    
    @property
    def plot(self):
        if not self.plot_available:
            raise Exception('ASFSBASStack.plot() requires additional dependencies: plotly and/or networkx')
        return self._plot
            
    
    def get_ref_stacks(self, stack_gdf: gpd.GeoDataFrame, season_bounds: tuple[int,int]) ->  gpd.GeoDataFrame:
        """
        Performs stack searches for any reference scenes that haven't had them yet, adding the results to the 'stack' column
        of self.sbas_stack.

        Merges self.sbas_stack (if it exists) with the input GeoDataFrame, which may contain additional ASFProducts from an
        expanded ASFSBASStack date range. An asf_search.stack_from_id search is performed on any scene in the resulting 
        GeoDataFrame whose 'stack' column doesn't contain results from a previous stack search. New results are saved to 
        the 'stack' column of the GeoDataFrame. 
        
        Args:
            stack_gdf (GeoDataFrame): a GeoDataFrame whose data is comprised of the .baseline and .properties attributes of
                the ASFProduct objects contained in an ASFSearchResults objects. The GeoDataFrame's geometry is from the 
                ASFProducts' .geometry attribute.
            season_bounds (tuple[int,int]): (Jan-2, Apr-1) == (1,91)

        Returns:
            A GeoDataFrame with the results of an asf.stack_from_id search for each scene it contains in the 'stack' column
            
        """
        if self._sbas_stack is not None: 
            # merge updated stack_gdf with existing _sbas_stack so we only need to make API calls on added scenes 
            stack_gdf = self._sbas_stack.merge(stack_gdf, on=['sceneName'], how='right')

            # clean up after the merge
            x_to_copy = ['insarNeighbors_x', 'stack_x']
            y_to_copy = [i for i in stack_gdf.columns if '_y' in i and 'insarNeighbors' not in i and 'stack' not in i]
            
            for y in y_to_copy:
                stack_gdf[y.split('_y')[0]] = stack_gdf[y]
            for x in x_to_copy:
                stack_gdf[x.split('_x')[0]] = stack_gdf[x]
                
            to_drop = [i for i in stack_gdf.columns if '_x' in i or '_y' in i]
            stack_gdf = stack_gdf.drop(columns=to_drop)

        args = asf.ASFSearchOptions(
            **{
                'start': self._start,
                'end': pd.Timestamp.now().strftime('%Y-%m-%d'),
                'season': season_bounds
            }
        )

        # update 'stack' with stack search results for ref scenes not previously searched
        stack_gdf['stack'] = stack_gdf.progress_apply(
            lambda row: (
                print(f"Downloading baseline stack for reference scene: {row['sceneName']}...") or 
                list(asf.stack_from_id(
                f"{row['sceneName']}-SLC" if 'BURST' not in row['sceneName'] else row['sceneName'], 
                args
            ))) if (not isinstance(row['stack'], list) and pd.isna(row['stack'])) else row['stack'],
            axis=1
        )
        return stack_gdf

        
    def centered_sublist(self, stack: list[asf.ASFProduct], target: int, length: int) -> list[asf.ASFProduct]:
        """
        Given an asf_search.stack_from_id search results list of ASFProducts and a target temporal baseline, returns a 
        sublist of a given length whose temporal baselines are centered around the target.

        Arguments:
            stack  (list[asf_search.ASFProduct]): The results of an asf_search.stack_from_id search
            target (int): The day of the year to center the sublist around
            length (int): The length of the sublist

        Returns:
            A sublist of the given length whose temporal baselines are centered around the target

        """
        temp_baselines = [i.properties['temporalBaseline'] for i in stack]
        if len(temp_baselines) == 0:
            return []
        closest_value = min(temp_baselines, key=lambda x: abs(x - target))
        target_index = temp_baselines.index(closest_value)
        half_length = length // 2
        
        start_index = max(0, target_index - half_length)
        end_index = min(len(stack), start_index + length)
        
        if end_index - start_index < length:
            start_index = max(0, end_index - length)
    
        return [stack[i] for i in range(start_index, end_index)]
    
    
    def baseline_stack_filter(self, stack: list[asf.ASFProduct], temporal_baseline_range: tuple[int, int]):
        """
        Takes an input asf_search.stack_from_id search results list and a range of temporal baselines. Returns a sublist
        of asf_search.ASFProducts whose temporal baselines fall within the given range and whose perpendicular baseline
        is less than or equal to self.perp_baseline.

        Arguments:
            stack (list[asf_search.ASFProduct]): asf_search.stack_from_id search results list
            temporal_baseline_range (tuple[int, int]): Range of temporal baselines by which to filter ASFProducts
        Returns:
            A sublist of asf_search.ASFProducts whose temporal baselines fall within the given range and whose 
            perpendicular baseline is less than or equal to self.perp_baseline

        """
        return [
            i for i in stack 
                if (
                    temporal_baseline_range[0] < i.properties['temporalBaseline'] <= temporal_baseline_range[1]
                    and
                    i.properties['perpendicularBaseline']
                    and
                    np.abs(i.properties['perpendicularBaseline']) <= self.perp_baseline
                    and
                    pd.Timestamp(i.properties['stopTime'], tz='UTC') <= pd.Timestamp(self._end, tz='UTC')
                )
            ]
    
    
    def get_seasonal_nearest_neighbors(self, stack_gdf: gpd.GeoDataFrame, ref_scene_name: str) -> list[asf.ASFProduct]:
        """
        Used to recalculate the SBAS stack pairs. 
        
        Given an sbas_stack GeoDataFrame with a complete 'stack' column and a reference scene ID that it contains
        (in the 'sceneName' column), returns a list of the secondary inSAR scenes for the given reference scene. 
        If the expected number of pairs cannot be made within the current season, 1-year pairs (+ or - the 
        temporal baseline) are attempted. The expected number of pairs for each reference scene is determined by
        `self.temporal_baseline // self.repeat_pass_freq`

        Arguments:
            stack_gdf (geopandas.GeoDataFrame): GeoDataFrame with a complete 'stack' column
            ref_scene_name (str): ID of the reference scene for whose secondary scenes we are searching 
            
        Returns:
            A list of secondary scenes for the provided reference scene that adheres to the currently defined 
            contraints of the SBAS stack
        
        """
        stack = stack_gdf.loc[stack_gdf.sceneName == ref_scene_name]['stack'].iloc[0]
    
        # create stack of in-season scenes, within baseline thresholds
        cur_season_stack = self.baseline_stack_filter(stack, (0, self.temporal_baseline))
    
        # find the number of expected neighbors 
        neighbor_len = self.temporal_baseline // self.repeat_pass_freq
    
        # return the current season's stack if enough neighbors were found
        if len(cur_season_stack) >= neighbor_len:
            return cur_season_stack
    
        # create stack of next-season scenes
        next_season_stack = self.baseline_stack_filter(stack, (365-self.temporal_baseline, 365+self.temporal_baseline))
    
        # find number remaining neighbors to identify
        neighbor_len = neighbor_len - len(cur_season_stack)
    
        # return any current season and next season scenes found
        return cur_season_stack + self.centered_sublist(next_season_stack, 365, neighbor_len)

    
    def get_insar_pairs(self) -> list[asf.ASFProduct]:
        """
        Useful when ordering an SBAS stack from ASF HyP3 On-Demand Processing
        
        If an ASFSBASStack.sbas_stack GeoDataFrame is available, generate a list tuples containing the reference and secondary
        scene name for each inSAR pair in the SBAS stack.
        
        Returns:
            A list tuples containing the reference and secondary scene name for each inSAR pair in the SBAS stack
        """
        
        return [
            (
                row['sceneName'], 
                neighbor.properties['sceneName']
            ) 
            for _, row in self.sbas_stack.iterrows() 
            for neighbor in row['insarNeighbors']
        ]

    
    def _plot(self):
        """
        Plot the SBAS stack

        """
        import plotly.graph_objects as go
        import networkx as nx
        G = nx.Graph()

        insar_node_pairs = [
            (
                row['stopTime'].split('T')[0], 
                neighbor.properties['stopTime'].split('T')[0], 
                {
                    'perp_bs': neighbor.properties['perpendicularBaseline'],
                    'temp_bs': neighbor.properties['temporalBaseline']
                }
            ) 
            for _, row in self.sbas_stack.iterrows() 
            for neighbor in row['insarNeighbors']
        ]
        
        G.add_edges_from(insar_node_pairs, data=True)
        
        scene_dates = {row['stopTime'].split('T')[0]: row['stopTime'].split('T')[0] for _, row in self._sbas_stack.iterrows()}
        nx.set_node_attributes(G, scene_dates, 'date')
        
        perp_bs = {row['stopTime'].split('T')[0]: row['perpendicularBaseline'] for _, row in self._sbas_stack.iterrows()}
        nx.set_node_attributes(G, perp_bs, 'perp_bs')
        
        node_positions = {node: datetime.strptime(data['date'], '%Y-%m-%d').timestamp() for node, data in G.nodes(data=True)}
        node_y_positions = {node: data['perp_bs'] for node, data in G.nodes(data=True)}
        
        node_x = [node_positions[node] for node in G.nodes()]
        node_y = [node_y_positions[node] for node in G.nodes()]
        node_text = [G.nodes[node]['date'] for node in G.nodes()]
        
        edge_x = []
        edge_y = []
        edge_text = []
        for edge in G.edges(data=True):
            x0 = node_positions[edge[0]]
            y0 = node_y_positions[edge[0]]
            x1 = node_positions[edge[1]]
            y1 = node_y_positions[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_text.append(f"{edge[0]} - {edge[1]}, perp baseline: {edge[2]['perp_bs']}, temp baseline: {edge[2]['temp_bs']}")
        
        start_date = min(scene_dates.values())
        end_date = max(scene_dates.values())
        date_range = pd.date_range(start=start_date, end=end_date, freq='MS').strftime('%Y-%m').tolist()
        date_range_ts = [datetime.strptime(date, '%Y-%m').timestamp() for date in date_range]
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=4, color='rgba(52, 114, 168, 0.7)'),
            mode='lines'
        )
        
        edge_hover_trace = go.Scatter(
            x=[(node_positions[edge[0]] + node_positions[edge[1]]) / 2 for edge in G.edges()],
            y=[(node_y_positions[edge[0]] + node_y_positions[edge[1]]) / 2 for edge in G.edges()],
            mode='markers',
            marker=dict(size=20, color='rgba(255, 255, 255, 0)'),
            hoverinfo='text',
            text=edge_text
        )
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            textposition="top center",
            marker=dict(size=15, color='rgba(32, 33, 32, 0.7)', line_width=0),
            hoverinfo='text',
            hovertext=node_text
        )
        
        def f_date(dash_date_str):
            return dash_date_str.replace("-", "/")
        
        fig = go.Figure(data=[edge_trace, edge_hover_trace, node_trace],
                        layout=go.Layout(
                            showlegend=False,
                            hovermode='closest',
                            height=800,
                            margin=dict(t=180),
                            xaxis=dict(
                                title='Acquisition Date',
                                tickvals=date_range_ts,
                                ticktext=date_range,
                                gridcolor='gray',
                                zerolinecolor='gray'
                            ),
                            yaxis=dict(
                                title='Perpendicular Baseline (m)',
                                gridcolor='gray',
                                zerolinecolor='gray'
                            ),
                            title=dict(
                                text=(
                                    '<b>Sentinel-1 Seasonal SBAS Stack</b><br>'
                                    f'Reference: {self._ref_scene_id}<br>'
                                    f'Temporal Bounds: {f_date(self._start)} - {f_date(self._end)}, Seasonal Bounds {f_date(self._season[0])} - {f_date(self._season[1])}<br>'
                                    f'Max Temporal Baseline: {self.temporal_baseline} days, Max Perpendicular Baseline: {self.perp_baseline}m<br>'
                                    f'Stack Size: {len(insar_node_pairs)} pairs from {len(scene_dates)} scenes<br>'
                                ),
                                y=0.95,
                                x=0.5,
                                xanchor='center',
                                yanchor='top',
                                font=dict(
                                    family="Helvetica, monospace",
                                    size=22,
                                )
                            ),
                            plot_bgcolor='white',
                            paper_bgcolor='lightgrey',
                        ))   
        fig.show()

    def build_sbas_stack(self) -> gpd.GeoDataFrame:
        """
        Builds or updates the ASFSBASStack.sbas_stack GeoDataFrame based on currently set stack parameters.

        Returns:
            The new or updated SBAS stack GeoDataFrame
        """
        if not self._ref_scene_id:
            raise(Exception('Stack "refSceneName" not set'))
        if not self._start:
            raise(Exception('Stack "start" not set'))
        if not self._end:
            raise(Exception('Stack "end" not set'))
        if pd.to_datetime(self._end) <= pd.to_datetime(self._start):
            raise(Exception(f'Stack end date is earlier than its start date. Correct to proceed.'))
        
        season_start_ts = pd.Timestamp(datetime.strptime(self._season[0], '%m-%d'), tz='UTC')
        season_start_day = season_start_ts.timetuple().tm_yday
        season_end_ts = pd.Timestamp(datetime.strptime(self._season[1], '%m-%d'), tz='UTC')
        season_end_day = season_end_ts.timetuple().tm_yday
        season = (season_start_day, season_end_day)
        
        args = asf.ASFSearchOptions(
            **{
                'start': self._start,
                'end': self._end,
                'season': season
            }
        )
        
        # get baseline stack from stack ref scene
        stack = asf.stack_from_id(self._ref_scene_id, args)

        # put the stack in a GeoDataFrame
        gdf = gpd.GeoDataFrame([s.properties|s.baseline for s in stack], geometry=[shape(s.geometry) for s in stack])
        gdf['insarNeighbors'] = [float('nan') for _ in range(len(gdf))]
        gdf['stack'] = [float('nan') for _ in range(len(gdf))]
        
        gdf = self.get_ref_stacks(gdf, season)
        gdf['insarNeighbors'] = gdf.apply(
            lambda row: self.get_seasonal_nearest_neighbors(gdf, row['sceneName']),
            axis=1
        )
        return gdf

 