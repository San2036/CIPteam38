// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FLRegistry {
    
    struct Node {
        uint256 id;
        uint256 reputation;
        bool isBanned;
        address nodeAddress;
    }
    
    mapping(address => Node) public nodes;
    address[] public nodeAddresses;
    
    string public globalModelHash;
    address public trustAgent;
    
    event NodeRegistered(address indexed nodeAddress, uint256 nodeId);
    event ModelUpdated(
        address indexed nodeAddress,
        string newHash,
        bytes proof,
        uint256 timestamp
    );
    event NodeSlashed(address indexed badNode, uint256 timestamp);
    
    modifier onlyTrustAgent() {
        require(msg.sender == trustAgent, "Only trust agent can call this function");
        _;
    }
    
    modifier notBanned() {
        require(!nodes[msg.sender].isBanned, "Node is banned");
        _;
    }
    
    constructor() {
        trustAgent = msg.sender;
        globalModelHash = "initial_model_hash";
    }
    
    function registerNode(uint256 nodeId) external {
        require(nodes[msg.sender].nodeAddress == address(0), "Node already registered");
        
        nodes[msg.sender] = Node({
            id: nodeId,
            reputation: 100,
            isBanned: false,
            nodeAddress: msg.sender
        });
        
        nodeAddresses.push(msg.sender);
        emit NodeRegistered(msg.sender, nodeId);
    }
    
    function uploadUpdate(string memory newHash, bytes memory proof) external notBanned {
        require(nodes[msg.sender].nodeAddress != address(0), "Node not registered");
        require(bytes(newHash).length > 0, "Hash cannot be empty");
        
        globalModelHash = newHash;
        
        emit ModelUpdated(msg.sender, newHash, proof, block.timestamp);
    }
    
    function slashNode(address badNode) external onlyTrustAgent {
        require(nodes[badNode].nodeAddress != address(0), "Node not found");
        
        nodes[badNode].reputation = 0;
        nodes[badNode].isBanned = true;
        
        emit NodeSlashed(badNode, block.timestamp);
    }
    
    function getNodeInfo(address nodeAddr) external view returns (Node memory) {
        return nodes[nodeAddr];
    }
    
    function getGlobalModelHash() external view returns (string memory) {
        return globalModelHash;
    }
    
    function getActiveNodeCount() external view returns (uint256) {
        uint256 count = 0;
        for (uint256 i = 0; i < nodeAddresses.length; i++) {
            if (!nodes[nodeAddresses[i]].isBanned) {
                count++;
            }
        }
        return count;
    }
}
