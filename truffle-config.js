module.exports = {
    networks: {
        development: {
            host: "127.0.0.1",     // Localheader
            port: 7545,            // Ganache GUI port
            network_id: "*",       // Any network (default: none)
        },
    },

    // Set default mocha options here, use special reporters etc.
    mocha: {
        // timeout: 100000
    },

    // Configure your compilers
    compilers: {
        solc: {
            version: "0.8.0",    // Fetch exact version from solc-bin (default: truffle's version)
        }
    },

    contracts_directory: "./blockchain",
    migrations_directory: "./blockchain/migrations",
    contracts_build_directory: "./build/contracts"
};
