from data_loader import fetch_data


def run():

    file_name = "default_D18_F40.tdms"
    directory = "data"
    channel_names = ('Index', 'FX_S1Plus2_COMP_T', 'FY_S1Plus2_COMP_T',
                     'FZ_S1Plus2_COMP_T', 'POSX_T', 'POSY_T', 'POSZ_T')

    fetch_data(channel_names, file_name, directory)
