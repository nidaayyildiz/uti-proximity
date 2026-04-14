from sdks.novavision.src.helper.package import PackageHelper
from components.Proximity.src.models.PackageModel import (
    PackageModel,
    PackageConfigs,
    ConfigExecutor,
    SocialGroupExecutor,
    SocialGroupResponse,
    SocialGroupOutputs,
    OutputGroups,
    OutputStats,
)


def build_response_social_group(context):
    output_groups = OutputGroups(value=context.groups_output)
    output_stats = OutputStats(value=context.stats_output)
    outputs = SocialGroupOutputs(outputGroups=output_groups, outputStats=output_stats)
    response = SocialGroupResponse(outputs=outputs)
    executor = SocialGroupExecutor(value=response)
    config_executor = ConfigExecutor(value=executor)
    package_configs = PackageConfigs(executor=config_executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=package_configs)
    package_model = package.build_model(context)
    return package_model