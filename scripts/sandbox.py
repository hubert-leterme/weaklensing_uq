import wlmmuq.data.cosmos as wlcosmos
import wlmmuq.data.kappatng as wlktng

imgsize = 384

cat_cosmos, _ = wlcosmos.cosmos_catalog()
data_dict = wlcosmos.get_data_from_cosmos(cat_cosmos, imgsize, wlktng.RESOLUTION)
