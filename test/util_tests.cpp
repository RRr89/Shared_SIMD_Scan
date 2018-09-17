#include "catch.hpp"
#include "util.hpp"

TEST_CASE("Find next multiple", "[util]")
{
    REQUIRE(next_multiple(0, 8) == 0);
    REQUIRE(next_multiple(5, 8) == 8);
    REQUIRE(next_multiple(8, 8) == 8);
    REQUIRE(next_multiple(15, 8) == 16);
    REQUIRE(next_multiple(16, 8) == 16);
    REQUIRE(next_multiple(17, 8) == 24);
    REQUIRE(next_multiple(17, 9) == 18);
}

TEST_CASE("Get bit in vector", "[util]")
{
    std::vector<uint8_t> vec{ 5, 5 };

    REQUIRE(get_bit(vec, 0) == true);
    REQUIRE(get_bit(vec, 1) == false);
    REQUIRE(get_bit(vec, 2) == true);
    REQUIRE(get_bit(vec, 3) == false);
    REQUIRE(get_bit(vec, 4) == false);
    REQUIRE(get_bit(vec, 5) == false);
    REQUIRE(get_bit(vec, 6) == false);
    REQUIRE(get_bit(vec, 7) == false);

    REQUIRE(get_bit(vec, 8) == true);
    REQUIRE(get_bit(vec, 9) == false);
    REQUIRE(get_bit(vec, 10) == true);
    REQUIRE(get_bit(vec, 11) == false);
    REQUIRE(get_bit(vec, 12) == false);
    REQUIRE(get_bit(vec, 13) == false);
    REQUIRE(get_bit(vec, 14) == false);
    REQUIRE(get_bit(vec, 15) == false);
}