/// Per-timestep board features fed to the model. Port of `snake_ai/features.py`.
///
/// Layout (16 floats):
///
///     [0:4]   distance to wall  up/down/left/right, normalized to [0, 1]
///     [4:6]   (dx, dy) to nearest food,           normalized to [-1, 1]
///     [6:10]  own direction, one-hot               (up, down, left, right)
///     [10:12] (dx, dy) to the opponent's head,     normalized to [-1, 1]
///     [12:16] opponent direction, one-hot
public enum FeatureExtractor {
    public static let featureDim = 16

    public static func features(of game: SnakeGame, for snakeID: Int) -> [Float] {
        let me = game.snakes[snakeID]
        let opp = game.snakes[1 - snakeID]
        let (hx, hy) = (Float(me.head.x), Float(me.head.y))
        let w = Float(game.width - 1)
        let h = Float(game.height - 1)

        // Distance to wall in each direction, normalized to [0, 1].
        let dWallUp = hy / h
        let dWallDown = (h - hy) / h
        let dWallLeft = hx / w
        let dWallRight = (w - hx) / w

        // Vector to food, normalized to [-1, 1].
        let dFoodX = (Float(game.food.x) - hx) / w
        let dFoodY = (Float(game.food.y) - hy) / h

        // Vector to the opponent's head. A dead opponent contributes zeros.
        let dUserX: Float, dUserY: Float, dirU: [Float]
        if opp.alive {
            dUserX = (Float(opp.head.x) - hx) / w
            dUserY = (Float(opp.head.y) - hy) / h
            dirU = opp.direction.oneHotEncoding
        } else {
            (dUserX, dUserY, dirU) = (0, 0, [0, 0, 0, 0])
        }

        var features = [Float]()
        features.reserveCapacity(featureDim)
        features += [dWallUp, dWallDown, dWallLeft, dWallRight]
        features += [dFoodX, dFoodY]
        features += me.direction.oneHotEncoding
        features += [dUserX, dUserY]
        features += dirU
        assert(features.count == featureDim)
        return features
    }
}
