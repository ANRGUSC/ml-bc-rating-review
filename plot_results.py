import plotly.express as px
import pathlib
import pandas as pd
import numpy as np

thisdir = pathlib.Path(__file__).parent.absolute()

def main():
    df = pd.read_csv(thisdir / 'sim.csv')

    df_final = df[df['step'] == 100]
    # take 'mode' and 'distance' columns and make a distance column for each mode (e.g. 'distance_est')
    df_final = df_final.pivot(
        index=['group_func', 'next_points_func', 'run'],
        columns='mode',
        values='distance'
    )
    df_final['improvement'] = df_final['no-est'] / df_final['est']
    df_final = df_final.reset_index()
    # print(df_final)
    # df_final = df_final.groupby(['group_func', 'next_points_func'])['improvement'].agg(['mean', 'min', 'max']).reset_index()
    
    # df_final['error_y'] = df_final['max'] - df_final['mean']
    # df_final['error_y_minus'] = df_final['mean'] - df_final['min']

    print(df_final)

    fig = px.box(
        df_final,
        x='group_func',
        y='improvement',
        # color='mode',
        facet_col='next_points_func',
        facet_col_wrap=2,
        template='plotly_white',
        # color='group_func',
    )
    # make gray
    # fig.update_traces(marker_color='gray')

    # Assuming 'thisdir' is a Path object
    fig.write_html(thisdir / 'sim.html')

if __name__ == '__main__':
    main()