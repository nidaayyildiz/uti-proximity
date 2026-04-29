from sdks.novavision.src.helper.package import PackageHelper
from components.Proximity.src.models.PackageModel import (
    PackageModel, PackageConfigs, ConfigExecutor,
    OutputGroups, ProximityOutputs, ProximityResponse, ProximityExecutor,
)


def build_response(context):
    output_groups   = OutputGroups(value=context.groups)
    outputs         = ProximityOutputs(outputGroups=output_groups)
    response        = ProximityResponse(outputs=outputs)
    executor        = ConfigExecutor(value=ProximityExecutor(value=response))
    package_configs = PackageConfigs(executor=executor)
    package         = PackageHelper(packageModel=PackageModel, packageConfigs=package_configs)
    return package.build_model(context)
