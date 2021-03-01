from matplotlib import cm
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
from skimage import transform

from . import etopo_kwargs#, extract_etopo
# from .. import grid
# from ..tools.misc import upscale_filter

FIGURE_SIZE_BASE = 10
FONT_SIZE = 16

def plot_terrain(
        z,
        ax,
        cax=None,
        fig=None,
        ls=None,
        vert_exag=None,
        blend_mode=None,
        cbar_orientation=None,
):
    # Plot
    if ls is not None:
        if vert_exag is None:
            vert_exag = 0.5
        if blend_mode is None:
            blend_mode = 'soft'
        im = ax.imshow(
            ls.shade(
                np.flipud(z),
                vert_exag=vert_exag,
                blend_mode=blend_mode,
                **etopo_kwargs
            ),
            zorder=0,
        )
    else:
        im = ax.imshow(
            np.flipud(z),
            zorder=0,
            **etopo_kwargs
        )

    # Colourbar
    if cax is not None:
        if fig is None:
            fig = plt.gcf()
        if cbar_orientation is None:
            cbar_orientation = 'horizontal'
        temp = ax.imshow(z, **etopo_kwargs)
        cbar = fig.colorbar(temp, cax=cax, orientation=cbar_orientation)
        temp.remove()

    if cax is None:
        return im
    else:
        return im, cbar


def plot_discharge(
        discharge,
        ax,
        cmap_discharge=None,
        norm=None,
        alpha=None,
        cax=None,
        fig=None,
        dmin=1.,
        dmax=1.e12,
        cbar_orientation=None,
):
    '''
    Plot discharge.
    Parameters:
        discharge
        ax
        cmap_discharge
        alpha: a string 'linear', 'log', 'ln' or a scalar or array
        fig
        cax
        dmin
        dmax
    '''
    if type(alpha) is str:
        if alpha not in ['linear', 'log', 'ln']:
            raise ValueError("Invalid value for 'alpha': {}".format(alpha))
        alphas = discharge.copy()
        alphas[alphas > dmax] = dmax
        alphas[alphas < dmin] = dmin
        if alpha == 'linear':
            pass
        elif alpha == 'log':
            alphas = np.log(alphas)
        elif alpha == 'ln':
            alphas = np.ln(alphas)
        alphas -= alphas.min()
        alphas /= alphas.max()
    else:
        alphas = alpha

    try:
        alphas = np.flipud(alphas)
    except ValueError:
        pass

    if cmap_discharge is None:
        cmap_discharge = 'Blues'

    try:
        if norm.lower() == 'log':
            norm = colors.LogNorm(vmin=dmin, vmax=dmax)
        else:
            norm = None
    except AttributeError:
        norm = None

    # Plot
    im = ax.imshow(
        np.flipud(discharge),
        norm=norm,
        cmap=cmap_discharge,
        alpha=alphas,
        zorder=10,
    )

    # Colourbar
    if cax is not None:
        if fig is None:
            fig = plt.gcf()
        if cbar_orientation is None:
            cbar_orientation = 'horizontal'
        temp = ax.imshow(discharge, norm=norm, cmap=cmap_discharge)
        cbar = fig.colorbar(temp, cax=cax, orientation=cbar_orientation)
        temp.remove()

    if cax is None:
        return im
    return im, cbar


def plot_erodep(
        dz,
        ax,
        cmap_erodep=None,
        cax=None,
        fig=None,
        dzmin=-2000,
        dzmax=2000,
        ls=None,
        z=None,
        vert_exag=None,
        blend_mode=None,
        cbar_orientation=None,
):
    if cmap_erodep is None:
        cmap_erodep = 'coolwarm'
    cmap_erodep = plt.cm.get_cmap(cmap_erodep)
    divnorm = colors.TwoSlopeNorm(
        vmin=dzmin,
        vcenter=0,
        vmax=dzmax,
    )

    # Plot
    if ls is not None:
        if z is None:
            z = dz
        if vert_exag is None:
            vert_exag = 0.5
        if blend_mode is None:
            blend_mode = 'soft'
        intensity = ls.hillshade(
            z,
            vert_exag=vert_exag,
            fraction=1.,
        )
        im = ax.imshow(
            np.flipud(dz),
            norm=divnorm,
            cmap=cmap_erodep,
            zorder=1,
        )
        ax.imshow(
            np.flipud(intensity),
            cmap='Greys',
            zorder=10,
            alpha=0.3,
        )
    else:
        im = ax.imshow(
            np.flipud(dz),
            cmap=cmap_erodep,
            zorder=0,
        )

    # Colourbar
    if cax is not None:
        if fig is None:
            fig = plt.gcf()
        if cbar_orientation is None:
            cbar_orientation = 'horizontal'
        temp = ax.imshow(dz, norm=divnorm, cmap=cmap_erodep)
        cbar = fig.colorbar(temp, cax=cax, orientation=cbar_orientation)
        temp.remove()

    if cax is None:
        return im
    return im, cbar


# def plot_diff(
#         input_topo,
#         step,
#         badlands_out,
#         etopo_filename,
#         llc,
#         urc,
#         output_filename_diff,
#         output_filename_comparison,
#         contour_interval=1000.,
#         filter_sigma=None,
# ):
#     regZ = grid.remap_TIN(
#         input_topo, step, badlands_out=badlands_out, verbose=False
#     )

#     etopo_Z = extract_etopo(etopo_filename, llc, urc, shape=regZ.shape)
#     if filter_sigma is not None:
#         diff = (
#             upscale_filter(regZ, sigma=filter_sigma)
#             - upscale_filter(etopo_Z, sigma=filter_sigma)
#         )
#     else:
#         diff = regZ - etopo_Z

#     cmap, divnorm = diverging_cmap(
#         cm.RdBu, vmin=diff.min(), vmax=diff.max(), vcenter=0,
#     )

#     # Plot difference
#     fig = plt.figure(figsize=(8, 12))
#     ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
#     im = ax.imshow(diff, cmap=cmap, norm=divnorm)
#     ax.contour(regZ, [0], linestyles='solid', linewidths=1.5, colors='k')
#     ax.set_ylim(0, diff.shape[0] - 1)

#     for i in [ax.xaxis, ax.yaxis]:
#         i.set_ticks([])

#     cax = fig.add_axes([0.1, 0.07, 0.8, 0.03])
#     fig.colorbar(im, cax=cax, orientation='horizontal')
#     cax.xaxis.set_tick_params(labelsize=13)
#     xlabel = r'$\mathrm{Z_{model} - Z_{ETOPO1}\;(m)}$'
#     cax.set_xlabel(xlabel, fontsize=20)

#     ax.set_title('Model v. ETOPO1 topography', fontsize=26)

#     fig.savefig(output_filename_diff, dpi=500, bbox_inches='tight')
#     plt.close(fig)

#     # Plot comparison
#     fig, axs = plt.subplots(1, 2, figsize=(16, 12))
#     ax1, ax2 = axs

#     contours = np.arange(-10000., max([regZ.max(), etopo_Z.max()]), 1000.)
#     contours_neg = np.flip(np.arange(
#         0., min([regZ.min(), etopo_Z.min()]), -1 * contour_interval
#     ))
#     contours_pos = np.arange(
#         contour_interval, max([regZ.max(), etopo_Z.max()]), contour_interval
#     )
#     contours = np.concatenate([contours_neg, contours_pos])

#     for arr, ax in zip([regZ, etopo_Z], axs):
#         if filter_sigma is not None:
#             arr = upscale_filter(arr, sigma=filter_sigma)
#         im = ax.imshow(arr, **etopo_kwargs)
#         ax.contour(
#             arr, [0], linewidths=1.5, linestyles='solid', colors='k'
#         )
#         ax.contour(
#             arr, contours,
#             linewidths=0.8, linestyles='solid', colors='k', alpha=0.5,
#         )
#         for i in [ax.xaxis, ax.yaxis]:
#             i.set_ticks([])
#         ax.set_ylim(0, arr.shape[0] - 1)

#     cax = fig.add_axes([0.2, 0.1, 0.6, 0.02])
#     fig.colorbar(im, cax=cax, orientation='horizontal')
#     cax.xaxis.set_tick_params(labelsize=13)
#     cax.set_xlabel(
#         'Elevation (m)\n{}m contours'.format(contour_interval),
#         fontsize=20,
#     )

#     ax1.set_title('Model topography', fontsize=26)
#     ax2.set_title('ETOPO1', fontsize=26)

#     fig.savefig(output_filename_comparison, dpi=500, bbox_inches='tight')
#     plt.close(fig)

#     return diff


def contour_elevation(
        z,
        ax,
        contour_interval=None,
        **kwargs
):
    d_contours = {}
    d_zeros = {}
    for key in kwargs:
        if '_' in key:
            if key.split('_')[0].lower() == 'contour':
                d_contours[key.split('_')[-1].lower()] = kwargs[key]
            elif key.split('_')[0].lower() == 'zero':
                d_zeros[key.split('_')[-1].lower()] = kwargs[key]
        else:
            d_contours[key] = kwargs[key]
            d_zeros[key] = kwargs[key]

    if contour_interval is None:
        contour_interval = 500

    contours = np.concatenate((
        np.flipud(
            np.arange(
                -1 * contour_interval,
                z.min() - contour_interval,
                -1 * contour_interval,
            )
        ),
        np.arange(
            0 + contour_interval,
            z.max() + contour_interval,
            contour_interval,
        )
    ))

    # Draw contours
    c0 = ax.contour(
        np.flipud(z),
        [0.0],
        **d_zeros,
    )
    c1 = ax.contour(
        np.flipud(z),
        contours,
        **d_contours,
    )

    return c0, c1


def plot_output(
        d,
        which='elevation',
        filename=None,
        time=None,
        hillshade=False,
        resize_factor=1,
        **kwargs
):
    '''
    Plot the output of the Badlands model.
    Parameters:
        d: the output from grid.remap_TIN
    '''
    try:
        which = which.lower()
    except AttributeError as e:
        raise TypeError("Invalid 'which' parameter: {}".format(which)) from e
    if which not in ['elevation', 'elev_discharge', 'erodep']:
        raise ValueError("Invalid 'which' parameter: {}".format(which))

    z = d['z']
    dz = d['cumdiff']
    discharge = d['discharge']
    discharge = np.nan_to_num(discharge, nan=1.0)

    # Resize if desired
    if resize_factor != 1:
        if 'resize_order' in kwargs:
            resize_order = kwargs['resize_order']
        else:
            resize_order = 3
        if resize_factor < 1:
            anti_aliasing = True
        else:
            anti_aliasing = False
        ny, nx = z.shape
        new_ny = int(np.around(ny * resize_factor))
        new_nx = int(np.around(nx * resize_factor))
        z = transform.resize(
            z,
            (new_ny, new_nx),
            order=resize_order,
            preserve_range=True,
            anti_aliasing=anti_aliasing,
        )
        ny, nx = dz.shape
        new_ny = int(np.around(ny * resize_factor))
        new_nx = int(np.around(nx * resize_factor))
        dz = transform.resize(
            dz,
            (new_ny, new_nx),
            order=resize_order,
            preserve_range=True,
            anti_aliasing=anti_aliasing,
        )
        ny, nx = discharge.shape
        new_ny = int(np.around(ny * resize_factor))
        new_nx = int(np.around(nx * resize_factor))
        discharge = transform.resize(
            discharge,
            (new_ny, new_nx),
            order=resize_order,
            preserve_range=True,
            anti_aliasing=anti_aliasing,
        )

    # Draw and configure figure and axes
    figsize = kwargs.get('figsize', (8, 12))
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])
    ax.tick_params(
        **{
            i: False for i in ['bottom', 'labelbottom', 'left', 'labelleft']
        }
    )
    if 'font_size' in kwargs:
        font_size = kwargs['font_size']
    else:
        font_size = FONT_SIZE
    title_size = int(font_size * 2.5)
    tick_size = int(font_size * 0.75)
    disp = 0.04 * (font_size / 16)
    width = 0.03 * (font_size / 16)
    cax1 = fig.add_axes([1 - (disp + width), 0.214, width, 0.572/2])

    # Hillshade parameters
    if hillshade:
        azdeg = kwargs.get('azdeg', 180 - 135)
        altdeg = kwargs.get('altdeg', 70)
        vert_exag = kwargs.get('vert_exag', 0.5)
        blend_mode = kwargs.get('blend_mode', 'soft')
        ls = colors.LightSource(azdeg=azdeg, altdeg=altdeg)
    else:
        azdeg, altdeg, vert_exag, blend_mode, ls = (None,
                                                    None,
                                                    None,
                                                    None,
                                                    None)

    if which == 'elev_discharge':
        cax2 = fig.add_axes([disp, 0.214, width, 0.572/2])
        # Discharge plotting parameters
        if 'dmax' in kwargs:
            dmax = kwargs['dmax']
        else:
            dmax = 1.e12
        if 'dmin' in kwargs:
            dmin = kwargs['dmin']
        else:
            dmin = 1.
        if 'cmap_discharge' in kwargs:
            cmap_discharge = kwargs['cmap_discharge']
        else:
            cmap_discharge = 'Blues'
        alphas = discharge.copy()
        alphas[alphas > dmax] = dmax
        alphas[alphas < dmin] = dmin
        alphas -= alphas.min()
        alphas /= alphas.max()

        # Plot discharge and colourbar
        plot_discharge(
            discharge=discharge,
            ax=ax,
            cmap_discharge=cmap_discharge,
            norm='log',
            alpha='linear',
            cax=cax2,
            fig=fig,
            dmin=dmin,
            dmax=dmax,
            cbar_orientation='vertical',
        )

    if which in ['elevation', 'elev_discharge']:
        # Plot elevation and colourbar
        plot_terrain(
            z,
            ax,
            fig=fig,
            cax=cax1,
            ls=ls,
            vert_exag=vert_exag,
            blend_mode=blend_mode,
            cbar_orientation='vertical',
        )

    if which == 'erodep':
        cmap_erodep = kwargs.get('cmap_erodep', 'coolwarm')
        dzmin = kwargs.get('dzmin', -2000)
        dzmax = kwargs.get('dzmax', 2000)
        plot_erodep(
            dz=dz,
            ax=ax,
            cmap_erodep=cmap_erodep,
            cax=cax1,
            fig=fig,
            z=z,
            ls=ls,
            dzmin=dzmin,
            dzmax=dzmax,
            vert_exag=vert_exag,
            blend_mode=blend_mode,
            cbar_orientation='vertical',
        )

    # Contour elevation
    contour_interval = kwargs.get('contour_interval', None)
    if contour_interval is not None:
        contour_interval = float(contour_interval)
        if contour_interval > 0:
            contour_interval = kwargs['contour_interval']
            d_contours = {
                i: kwargs[i] for i in kwargs
                if '_' in i
                and (i[:4] == 'zero' or i[:7] == 'contour')
                and i != 'contour_interval'
            }
            if 'contour_linestyles' not in kwargs:
                d_contours['contour_linestyles'] = 'solid'
            if 'contour_linewidths' not in kwargs:
                d_contours['contour_linewidths'] = 0.7
            if 'contour_colors' not in kwargs:
                if which != 'erodep':
                    d_contours['contour_colors'] = 'white'
                else:
                    d_contours['contour_colors'] = 'grey'
            if 'zero_linestyles' not in kwargs:
                d_contours['zero_linestyles'] = None
            if 'zero_linewidths' not in kwargs:
                d_contours['zero_linewidths'] = None
            if 'zero_colors' not in kwargs:
                d_contours['zero_colors'] = 'k'
            contour_elevation(
                z=z,
                ax=ax,
                contour_interval=contour_interval,
                zero_zorder=50,
                contour_zorder=50,
                **d_contours,
            )
        elif contour_interval == 0:
            ax.contour(np.flipud(z),
                       [0.0],
                       linewidths=1.4,
                       linestyles='solid',
                       colors='black')
        else:
            raise ValueError(
                'contour_interval < 0: {}'.format(contour_interval)
            )

    # Adjust colourbars
    if which == 'elev_discharge':
        caxs = [cax1, cax2]
    else:
        caxs = [cax1]
    for cax in caxs:
        cax.tick_params(
            **{
                i: False for i in [
                    'bottom',
                    'labelbottom',
                    'right',
                    'labelright',
                    'top',
                    'labeltop',
                ]
            }
        )
        cax.tick_params(
            **{
                i: True for i in ['left', 'labelleft']
            }
        )
        cax.yaxis.set_tick_params(labelsize=tick_size)
        cax.yaxis.set_label_coords(0.5, 1.2 * (font_size / 16))
    cax1.tick_params(
        **{
            'left': False,
            'labelleft': False,
            'right': True,
            'labelright': True,
        }
    )

    # Add and configure labels
    if 'title' in kwargs:
        title = kwargs['title']
    elif time is not None:
        time_decimals = kwargs.get('time_decimals', 1)
        title = ('{:.' + str(time_decimals) + 'f} Ma').format(time)
    else:
        title = None
    title = fig.suptitle(
        title,
        y=0.2-(0.03*font_size/16),
        fontsize=title_size
    )

    if which in ['elevation', 'elev_discharge']:
        if contour_interval is not None:
            cbarlabel1 = (
                'Elevation (m)\n({}m contours)'.format(contour_interval)
            )
        else:
            cbarlabel1 = 'Elevation (m)'
        cbarlabel1 = cax1.set_ylabel(
            cbarlabel1,
            fontsize=font_size,
            rotation=0,
        )
    elif which == 'erodep':
        cbarlabel1 = '\n' + r'$\mathrm{\Delta Z (m)}$'
        cbarlabel1 = cax1.set_ylabel(
            cbarlabel1,
            fontsize=font_size,
            rotation=0,
        )

    if which == 'elev_discharge':
        cbarlabel2 = r'Discharge ($\mathrm{m^3\,{yr}^{-1}}$)'
        cbarlabel2 = cax2.set_ylabel(
            cbarlabel2,
            fontsize=font_size,
            rotation=0,
        )

    # Save to file
    if filename is not None:
        dpi = kwargs.get('dpi', 300)
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        plt.close(fig)


def plot_catchment(
        catchment_mask,
        elevation_map=None,
        discharge_map=None,
        delta_mask=None,
        output_filename=None,
        discharge_vmin=1.,
        discharge_vmax=1.e12,
        catchment_kwargs=None,
        elevation_kwargs=None,
        coastline_kwargs=None,
        discharge_kwargs=None,
        delta_kwargs=None,
        title=None,
        font_size=20,
        output_dpi=400,
        clf=True,
):
    tick_size = font_size * 0.75
    title_size = font_size * 1.5

    ny, nx = catchment_mask.shape
    figsize_factor = 10
    figsize = (figsize_factor, figsize_factor * (ny / nx))
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set_ylim(0, catchment_mask.shape[0] - 1)

    if elevation_map is not None:
        if elevation_kwargs is None:
            elevation_kwargs = etopo_kwargs
        im_elevation = ax.imshow(elevation_map, **elevation_kwargs)
        if coastline_kwargs is None:
            coastline_kwargs = {
                'colors': 'k',
                'linewidths': 1.5,
            }
        ax.contour(elevation_map, [0], **coastline_kwargs)

    if discharge_map is not None:
        if discharge_kwargs is None:
            alphas = discharge_map.copy()
            alphas[np.isnan(alphas)] = 0
            alphas[alphas < discharge_vmin] = discharge_vmin
            alphas[alphas > discharge_vmax] = discharge_vmax
            alphas -= alphas.min()
            alphas /= alphas.max()
            norm = colors.LogNorm(vmin=discharge_vmin, vmax=discharge_vmax)
            discharge_kwargs = {
                'cmap': 'Blues',
                'alpha': alphas,
                'norm': norm,
            }
        ax.imshow(discharge_map, **discharge_kwargs)
        temp = ax.imshow(
            discharge_map,
            cmap=discharge_kwargs.get('cmap', None),
            norm=discharge_kwargs.get('norm', None),
        )

    if catchment_kwargs is None:
        catchment_kwargs = {
            'colors': 'b',
            'linewidths': 1.5,
        }
    ax.contour(
        catchment_mask, [1.e-3], **catchment_kwargs
    )

    if delta_mask is not None:
        if delta_kwargs is None:
            delta_kwargs = {
                'colors': 'r',
                'linewidths': 1.0,
                'alpha': 0.5,
            }
        ax.contour(
            delta_mask, [1.e-3], **delta_kwargs
        )

    if elevation_map is not None:
        cax_elevation = fig.add_axes([0.1, 0.04, 0.8, 0.03])
        fig.colorbar(
            im_elevation, cax=cax_elevation, orientation='horizontal',
        )
        for xy in [cax_elevation.xaxis, cax_elevation.yaxis]:
            xy.set_tick_params(labelsize=tick_size)
        cax_elevation.set_xlabel('Elevation (m)', fontsize=font_size)
    if discharge_map is not None:
        cax_discharge = fig.add_axes([0.93, 0.1, 0.03, 0.8])
        fig.colorbar(
            temp, cax=cax_discharge, orientation='vertical',
        )
        temp.remove()
        for xy in [cax_discharge.xaxis, cax_discharge.yaxis]:
            xy.set_tick_params(labelsize=tick_size)
        cax_discharge.set_ylabel(
            r'Discharge ($\mathrm{m^3 \, {yr}^{-1}}$)',
            fontsize=font_size,
        )

    ax.set_xticks([])
    ax.set_yticks([])

    if title is not None:
        ax.set_title(title, fontsize=title_size)

    if output_filename is not None:
        fig.savefig(
            output_filename,
            dpi=output_dpi,
            bbox_inches='tight',
        )

    if clf:
        plt.close(fig)
    return fig


def plot_delta_erodep(
        erodep,
        llc,
        urc,
        z=None,
        output_filename=None,
        **kwargs
):
    '''
    Plot the delta bounding box, along with erosion/deposition.
    ------------------------------------------------------------------
    Parameters:
        erodep: erosion/deposition array
        llc: (x, y) coordinate pair of the delta bounding box
            lower-left corner. Can also be an (i, j) coordinate
            pair if the kwarg order='ij' is supplied.
        urc: (x, y) or (i, j) coordinate pair of the delta
            bounding box upper-right corner.
        z: elevation array (optional). If provided, coastlines
            will be drawn.
            (default None)
        output_filename: filename to save the figure (optional).
            If provided, the figure will be closed after saving.
            (default None)
        **kwargs: valid kwargs include 'font_size', 'order',
            'cmap', 'vmin', 'vmax', 'figsize', and 'title'.
    Returns:
        fig: the figure object containing the plot.
    '''
    # Make sure erodep and x have the same dimensions
    ny, nx = erodep.shape
    if z is not None:
        if z.shape != erodep.shape:
            raise ValueError('erodep and z must have the same shape')
    font_size = kwargs.get('font_size', FONT_SIZE)
    title_size = font_size * 1.6
    tick_size = font_size * 0.8

    # Copy data to make sure no changes are made
    # to the original
    erodep = erodep.copy()
    if z is not None:
        z = z.copy()
    erodep[np.isnan(erodep)] = 0.
    z[np.isnan(z)] = 0.
    # Parameters for plotting
    order = kwargs.get('order', 'xy')
    if order == 'ij':
        llc = llc[::-1]
        urc = urc[::-1]
    contour_interval = kwargs.get('contour_interval', 500.)
    cmap = kwargs.get('cmap', 'RdBu_r')
    vmin = kwargs.get('vmin', -1 * np.abs(erodep).max())
    vmax = kwargs.get('vmax', np.abs(erodep).max())
    image_kwargs = {
        'cmap': cmap,
        'vmin': vmin,
        'vmax': vmax,
    }
    coastline_kwargs = {
        'linewidth': 1.5,
        'colors': 'black',
        'linestyles': 'solid',
    }
    contours = np.concatenate((
        np.flip(
            np.arange(
                -1 * contour_interval,
                erodep.min() - contour_interval,
                -1 * contour_interval,
            )
        ),
        np.arange(
            contour_interval,
            erodep.max() + contour_interval,
            contour_interval,
        ),
    ))
    if cmap == 'RdBu_r':
        colours = [
            'blue' if i < 0 else 'red' for i in contours
        ]
    elif cmap == 'RdBu':
        colours = [
            'red' if i < 0 else 'blue' for i in contours
        ]
    else:
        colours = 'black'
    contour_kwargs = {
        'linewidth': 1.0,
        'colors': colours,
        'alpha': 0.2,
        'linestyles': 'solid',
    }
    box_kwargs = {
        'linewidth': 2.0,
        'color': 'red',
    }

    figure_size = kwargs.get('figsize', FIGURE_SIZE_BASE)
    fig = plt.figure(figsize=(figure_size, figure_size * (ny / nx)))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    im = ax.imshow(erodep, **image_kwargs)
    ax.contour(z, [0], **coastline_kwargs)
    ax.contour(erodep, contours, **contour_kwargs)
    ax.plot(
        [llc[0], urc[0], urc[0], llc[0], llc[0]],
        [llc[1], llc[1], urc[1], urc[1], llc[1]],
        **box_kwargs
    )

    ax.set_ylim(0, ny - 1)
    ax.set_xticks([])
    ax.set_yticks([])

    cax = fig.add_axes([0.1, 0.05, 0.8, 0.03])
    fig.colorbar(im, cax=cax, orientation='horizontal')
    cax.set_xlabel('Cumulative erosion/deposition (m)', fontsize=font_size)
    cax.xaxis.set_tick_params(labelsize=tick_size)

    title = kwargs.get('title', None)
    if title is not None:
        ax.set_title(title, fontsize=title_size)
    if output_filename is not None:
        fig.savefig(output_filename, dpi=400, bbox_inches='tight')
        plt.close(fig)
    else:
        fig.show()
    return fig


def diverging_cmap(
        cmap,
        vmin,
        vmax,
        vcenter=0,
        cmap_center=0.5,
        name='custom_cmap'
):
    '''
    Create a diverging colour map with a two-slope normalisation
    -------------------------------------------------------------
    Parameters:
        cmap: a matplotlib.colors.Colormap instance, or the name of a valid
            matplotlib colour map (i.e. valid input for matplotlib.cm.get_cmap)
        vmin: the minimum of the data
        vmax: the maximum of the data
        vcenter: the centre for the diverging colour map
            (default 0)
        cmap_center: the centre of cmap (usually 0.5)
            (default 0.5)
        name: the name to be given to the newly-created colour map
            (default 'custom_cmap')
    Returns:
        cmap_out: the new diverging colour map
        divnorm: the two-slope diverging normalisation
            (to be passed to plotting functions; e.g. norm=divnorm)
    '''
    if not isinstance(cmap, colors.Colormap):
        cmap = cm.get_cmap(name=cmap)

    colours_neg = cmap(np.linspace(0., cmap_center, 256))
    colours_pos = cmap(np.linspace(cmap_center, 1., 256))
    colours = np.vstack((colours_neg, colours_pos))
    divnorm = colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=vcenter)
    cmap_out = colors.LinearSegmentedColormap.from_list(name, colours)

    return cmap_out, divnorm
