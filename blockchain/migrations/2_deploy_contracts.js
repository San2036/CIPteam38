const FLRegistry = artifacts.require("FLRegistry");

module.exports = function (deployer) {
    deployer.deploy(FLRegistry);
};
