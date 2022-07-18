import pandas as pd
import numpy as np
import plotly.express as px
import folium
import streamlit as st
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster

#Configs
pd.set_option('display.float_format', lambda x: '%.2f' % x)
st.set_page_config(layout='wide')
@st.cache(allow_output_mutation=True)

#load
def get_data(path):
    data = pd.read_csv(path)
    return data

#Data Overview
def data_overview(data):
    st.title('BI Insights - House Sales in King County')
    st.header('Data Overview')
    st.subheader(f'Lines / Columns {data.shape}')
    st.dataframe(data.head())


#Commercial viability
def commercially_viable_buy(data):
    # Median price by zipcode
    df = data[['price', 'zipcode']].groupby('zipcode').median().reset_index()
    df = df.sort_values('price', axis=0)
    df2 = pd.merge(data, df, on='zipcode', how='inner')
    df2 = df2.rename(columns={'price_x': 'price', 'price_y': 'price_mean'}, inplace=False)

    # Size categorization by zipcode median
    quantiles = np.percentile(df2['sqft_living'], [25, 50, 75], interpolation='midpoint')
    df2['size_type'] = df2['sqft_living'].apply(lambda x: 1 if x <= int(quantiles[0]) else
                                                          2 if x <= int(quantiles[1]) and x > int(quantiles[0]) else
                                                          3 if x <= int(quantiles[2]) and x > int(quantiles[1]) else
                                                          4 if x > int(quantiles[2]) else 'na')
    df3 = df2[['zipcode', 'size_type']].groupby('zipcode').median().reset_index()
    df2 = pd.merge(df2, df3, on='zipcode', how='inner')
    df2 = df2.rename(columns={'size_type_y': 'avg_size', 'size_type_x': 'size_type'}, inplace=False)

    # Condition categorization
    df2['condition_type'] = df2['condition'].apply(lambda x: 'bad' if x <= 2 else
                                                             'regular' if x >= 3 and x < 5 else
                                                             'good' if x >= 5 else 'na')

    # Commercial viability avaliation
    df4 = df2
    df2['commercially_viable'] = df2[(df2['price'] < df2['price_mean'])
                                     & (df2['size_type'] > df2['avg_size'])
                                     & (df2['condition_type'] == 'good')]['id']
    df2['commercially_viable'] = df2['commercially_viable'].apply(lambda x: 'yes' if x > 0 else 'no')
    df3 = df2.loc[df2['commercially_viable'] == 'yes']

    # df4 used as base reference for price difference
    df4['commercially_viable'] = df2[(df2['price'] > df2['price_mean'])
                                     & (df2['size_type'] > df2['avg_size'])
                                     & (df2['condition_type'] == 'good')]['id']
    df4['commercially_viable'] = df4['commercially_viable'].apply(lambda x: 'yes' if x > 0 else 'no')
    df3['estimate_profit_buy'] = round((df3['price'].apply(lambda x: df4['price'].mean() - x)), 2)

    # Estimate profit of buying all houses
    profit = round(sum(df3['price'].apply(lambda x: df4['price'].mean() - x).tolist()), 2)

    # Scatter Map
    density_map = folium.Map(location=[data['lat'].mean(),
                                       data['long'].mean()],
                                       default_zoom_start='15')
    marker_cluster = MarkerCluster().add_to(density_map)
    for name, row in df3.iterrows():
        folium.Marker([row['lat'], row['long']]).add_to(marker_cluster)

    # Streamlit output
    st.title('Commercial viability analysis (Buy)')
    st.subheader(f'Total estimate profit ${profit}')
    folium_static(density_map)

    return df3


def selling_time(data):
    df1 = data
    df1['month'] = pd.DatetimeIndex(df1['date']).month
    df1['season'] = df1['month'].apply(lambda x: 'spring' if x >= 3 and x <= 5 else
                                                 'summer' if x >= 6 and x <= 8 else
                                                 'autumn' if x >= 9 and x <= 11 else 'winter')
    df2 = df1[['price', 'zipcode']].groupby('zipcode').median().reset_index()
    df3 = pd.merge(df1, df2, on='zipcode', how='inner')
    df3 = df3.rename(columns={'price_x': 'price', 'price_y': 'price_median_zipcode'}, inplace=False)

    def selling_price(season, price):
        if 'summer' in season:
            selling_price = price + price * 0.3
        elif 'autumn' in season:
            selling_price = price + price * 0.3
        elif 'spring' in season:
            selling_price = price + price * 0.3
        else:
            selling_price = price + price * 0.1
        return round(selling_price, 2)

    def selling_profit(price, selling_price):
        selling_profit = selling_price - price
        return selling_profit

    df3['selling_price'] = df3.apply(lambda x: selling_price(x['season'], x['price']), axis=1)
    df3['selling_profit'] = df3.apply(lambda x: selling_profit(x['price'], x['selling_price']), axis=1)
    profit = df3['selling_profit'].sum()

    st.title('Commercial viability analysis (Sell)')
    st.subheader(f'Total estimate profit ${profit}')
    st.dataframe(df3.head())

    return df3


#Hypothesis Testing

# Houses that have waterfront are 30% more expensive
def waterview_price_difference(data):
    waterview = data.loc[data['waterfront'] == 1]
    waterview_mean = waterview['price'].mean()
    not_waterview = data.loc[data['waterfront'] == 0]
    not_waterview_mean = not_waterview['price'].mean()
    percentage = round((waterview_mean - not_waterview_mean) * 100 / waterview_mean)
    d1 = data
    d1['is_waterfront'] = d1['waterfront'].apply(lambda x: 'Houses with waterfront' if x == 1
    else 'Houses without waterfront')
    waterfront = d1[['price', 'is_waterfront']].groupby('is_waterfront').mean().reset_index()
    fig_1 = px.bar(waterfront, x='is_waterfront', y='price', text_auto=True)
    st.title('Houses with waterfront are 30% more expensive')
    st.subheader(f'Houses with waterfront are {percentage}% more expensive than houses without it.')
    st.plotly_chart(fig_1, use_container_width=True)
    return None


# Houses that were built before 1955 are 50% cheaper in average
def before_1955(data):
    before1955_mean = data.loc[data['yr_built'] < 1955]['price'].mean()
    after1955 = data.loc[data['yr_built'] >= 1955]
    after1955_mean = after1955['price'].mean()
    percentage = round((after1955_mean - before1955_mean) * 100 / after1955_mean)
    d1 = data
    d1['1955'] = d1['yr_built'].apply(lambda x: 'Built before 1955' if x < 1955
    else 'Built after 1955')
    in_1955 = d1[['price', '1955']].groupby('1955').mean().reset_index()
    fig_1 = px.bar(in_1955, x='1955', y='price', text_auto=True)
    st.title('Average price difference between Houses that were built before 1955 and after it')
    st.subheader(f'Houses built before 1955 are {percentage}% cheaper than the houses built after that in average')
    st.plotly_chart(fig_1, use_container_width=True)
    return None


# Houses that have a basement are 40% larger
def basement_size(data):
    basement_houses = data.loc[data['sqft_basement'] != 0]
    basement_houses_mean = basement_houses['sqft_living'].mean()
    no_basement_houses = data.loc[data['sqft_basement'] == 0].mean()
    no_basement_houses_mean = no_basement_houses['sqft_living'].mean()
    percentage = round((basement_houses_mean - no_basement_houses_mean) * 100 / basement_houses_mean)
    d1 = data
    d1['basements'] = d1['sqft_basement'].apply(lambda x: 'Houses with basement' if x > 0
                                                           else 'Houses without basement')
    in_basements = d1[['sqft_living', 'basements']].groupby('basements').mean().reset_index()
    fig_1 = px.bar(in_basements, x='basements', y='sqft_living', text_auto=True)
    st.title('Houses with basements tend to have a large built living area than houses without it')
    st.subheader(f'Houses with basements are {percentage}% bigger')
    st.plotly_chart(fig_1, use_container_width=True)
    return None


# By mean values, prices of houses rises 10% each year
def yoy_price(data):
    data['date'] = pd.to_datetime(data['date'])
    data['year'] = pd.DatetimeIndex(data['date']).year
    mean_2014 = data.loc[data['year'] == 2014]['price'].mean()
    mean_2015 = data.loc[data['year'] == 2015]['price'].mean()
    percentage = round((mean_2015 - mean_2014) * 100 / mean_2015)
    d1 = data
    in_year = d1[['price', 'year']].groupby('year').mean().reset_index()
    fig_1 = px.bar(in_year, x='year', y='price', text_auto=True)
    st.title('By mean values, prices of houses tend to rise 10% each year')
    st.subheader(f'The mean price of the houses rose by {percentage}% from 2014 to 2015')
    st.plotly_chart(fig_1, use_container_width=True)
    return None


# The average MOM increase of price of houses with three or more bathrooms is 5%
def mom_bathrooms(data):
    data['date'] = pd.to_datetime(data['date'])
    data['year_month'] = data['date'].dt.to_period('M')
    df1 = data.loc[data['bathrooms'] >= 3]
    df1 = df1[['year_month', 'price']].groupby(['year_month']).mean().sort_values(by='year_month',
                                                                                  ascending=True).reset_index()
    prices = df1['price'].astype('int64').tolist()
    percentage = 0
    counter = 0
    while counter < len(prices):
        if len(prices) % 2 == 0:
            percentage += int((prices[counter] - prices[counter + 1]) * 100 / prices[counter])
            counter += 2
        else:
            prices.pop()
    percentage_mean = percentage / len(prices)
    df1['year_month'] = df1['year_month'].astype(str)
    fig_1 = px.line(df1, x='year_month', y='price')
    st.title('Month Over Month value increase percentage of houses with three or more bathrooms')
    st.subheader(f'The mean diference in price month by month of houses with more than three bathrooms is {percentage_mean}%')
    st.plotly_chart(fig_1, use_container_width=True)
    return None


# Houses with one room represent 15% of all the houses built after the year of 2010
def houses_1_room(data):
    df1 = data.loc[data['yr_built'] >= 2010]
    one_room = len(df1.loc[data['bedrooms'] == 1]) / len(df1.loc[data['bedrooms'] != 1]) * 100
    pie_data = {'labels': ['one_room', 'more_rooms'],
                'values': [len(df1.loc[df1['bedrooms'] == 1]),
                           len(df1.loc[df1['bedrooms'] != 1])]}
    pie_df = pd.DataFrame(data=pie_data)
    fig_1 = px.pie(pie_df, values='values', names='labels')
    st.title('Houses with one room represent 15% of all the houses built after the year of 2010')
    st.subheader(f'Houses with only one bedroom represent {one_room:.2f}% of all houses built after the year of 2010')
    st.plotly_chart(fig_1, use_container_width=True)


# Bigger houses are 50% more expensive than smaller ones
def big_houses(data):
    df1 = data
    quantiles = np.percentile(data['sqft_living'], [25, 50, 75], interpolation='midpoint')
    df1['size_type'] = data['sqft_living'].apply(lambda x: 1 if x <= int(quantiles[0]) else
                                                 2 if x <= int(quantiles[1]) and x > int(quantiles[0]) else
                                                 3 if x <= int(quantiles[2]) and x > int(quantiles[1]) else
                                                 4 if x > int(quantiles[2]) else 'na')
    mean_price_big = round(df1.loc[df1['size_type'] == 4]['price'].mean())
    mean_price_not_big = round(df1.loc[df1['size_type'] != 4]['price'].mean())
    percentage = round((mean_price_big - mean_price_not_big) * 100 / mean_price_big)
    bar_data = {'labels': ['bigger_houses', 'smaller_houses'],
                'values': [round(df1.loc[df1['size_type'] == 4]['price'].mean()),
                           round(df1.loc[df1['size_type'] != 4]['price'].mean())]}
    bar_df = pd.DataFrame(data=bar_data)
    fig_1 = px.bar(bar_df, x='labels', y='values', text_auto=True)
    st.title('Bigger houses are 50% more expensive than smaller ones')
    st.subheader(f'Houses on the higher end of size are {percentage}% more expensive by mean values than smaller houses')
    st.plotly_chart(fig_1, use_container_width=True)
    return None


# Houses that have a good condition are 30% more expensive
def good_condition(data):
    df1 = data
    df1['condition_type'] = df1['condition'].apply(lambda x: 'bad' if x <= 2 else
                                                             'regular' if x >= 3 and x < 5 else
                                                             'good' if x >= 5 else 'na')
    mean_price_good = round(df1.loc[df1['condition_type'] == 'good']['price'].mean())
    mean_price_not_good = round(df1.loc[df1['condition_type'] != 'good']['price'].mean())
    percentage = round((mean_price_good - mean_price_not_good) * 100 / mean_price_good)
    bar_data = {'labels': ['worst_condition', 'good_condition'],
                'values': [round(df1.loc[df1['condition'] < 5]['price'].mean()),
                           round(df1.loc[df1['condition'] >= 5]['price'].mean())]}
    bar_df = pd.DataFrame(data=bar_data)
    fig_1 = px.bar(bar_df, x='labels', y='values', text_auto=True)
    st.title('Houses that have a good condition are 30% more expensive')
    st.subheader(f'Houses with a good condition are {percentage}% more expensive')
    st.plotly_chart(fig_1, use_container_width=True)
    return None


# Houses with big lots are 20% more expensive
def big_lot(data):
    df1 = data
    quantiles = np.percentile(df1['sqft_lot'], [25, 50, 75], interpolation='midpoint')
    df1['lot_type'] = df1['sqft_lot'].apply(lambda x: 1 if x <= int(quantiles[0]) else
    2 if x <= int(quantiles[1]) and x > int(quantiles[0]) else
    3 if x <= int(quantiles[2]) and x > int(quantiles[1]) else
    4 if x > int(quantiles[2]) else 'na')
    lot_size_big = round(df1.loc[df1['lot_type'] == 4]['price'].mean())
    lot_size_not_big = round(df1.loc[df1['lot_type'] != 4]['price'].mean())
    percentage = round((lot_size_big - lot_size_not_big) * 100 / lot_size_big)
    bar_data = {'labels': ['bigger_houses', 'smaller_houses'],
                'values': [round(df1.loc[df1['lot_type'] >= 4]['price'].mean()),
                           round(df1.loc[df1['lot_type'] < 4]['price'].mean())]}
    bar_df = pd.DataFrame(data=bar_data)
    fig_1 = px.bar(bar_df, x='labels', y='values', text_auto=True)
    st.title('Houses with big lots are 20% more expensive')
    st.subheader(f'Houses with bigger lots are {percentage}% more expensive')
    st.plotly_chart(fig_1, use_container_width=True)
    return None


# Houses renovated within the last 5 years are more likely to recieve a grade of more than 10
def renovated_10(data):
    more_10 = data.loc[(data['grade'] >= 10) & (data['yr_renovated'] >= 2013)]['grade'].mean()
    less_10 = data.loc[(data['grade'] >= 10) & (data['yr_renovated'] < 2013)]['grade'].mean()
    percentage = (more_10 - less_10) * 100 / more_10
    bar_data = {'labels': ['Renovated after 2013', 'Renovated before 2013'],
                'values': [more_10, less_10]}
    bar_df = pd.DataFrame(data=bar_data)
    fig_1 = px.bar(bar_df, x='labels', y='values', text_auto=True)
    st.title('Houses renovated within the last 5 years are more likely to recieve a grade of more than 10')
    st.subheader(f'Houses renovated within the last 5 years are {percentage:.2f}% more likely to recieve a grade of 10 or more')
    st.plotly_chart(fig_1, use_container_width=True)
    return None

if __name__ == '__main__':
    path = 'kc_house_data_2.csv'
    data = get_data(path)
    data_overview(data)
    df_buy = commercially_viable_buy(data)
    df_sell = selling_time(data)
    waterview_price_difference(data)
    before_1955(data)
    basement_size(data)
    yoy_price(data)
    mom_bathrooms(data)
    houses_1_room(data)
    big_houses(data)
    good_condition(data)
    big_lot(data)
    renovated_10(data)
